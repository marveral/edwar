from utils import frequency_conversion, butter_lowpass_filter
from parser import Parser
import numpy as np 
import pandas as pd 
import pywt
from .eda_explorer.EDA_Artifact_Detection_Script import classifyEpochs, getSVMFeatures, getWaveletData, getFeatures

class EDAparser(Parser):
    def __init__(self):
        super().__init__(inputs={'EDA': 'uS'},  # optional_inputs={'ACCx': 'g', 'ACCy': 'g', 'ACCz': 'g'},
                         outputs={'EDA': 'uS'})

    def run(self, data, fs, classifier):
        self.check_input(self.__class__.__name__, data, self.inputs, self.optional_inputs)
        if not isinstance(classifier, list):
            raise TypeError('EDA_CLASSIFIER must be a list')
        if [c in ['Binary','Multiclass'] for c in classifier]:
            raise ValueError('EDA_CLASSIFIER must be Binary or Multiclass')
        eda_processed = self.process_eda( data, fs, classifier)
        return eda_processed
    
    def process_eda(self, data, fs, classifier=None):
        """ Method to process EDA signal. 
        1. Low pass filter of order 6 at 1Hz
        2. Artifacts are detected 
        3. Artifacts are corrected using wavelet thresholding
        4. Frequency conversion to 2Hz for faster converge of cvxEDA algorithm. 
        5. Normalization of the signals.

        Args:
            data (_type_): _description_
            fs (_type_): _description_
            classifier (_type_, optional): _description_. Defaults to None.

        Returns:
            _type_: _description_
        """
        data = frequency_conversion(data, f=8)
        if classifier is None:
            classifier = ['Binary']
        data['filtered_eda'] = butter_lowpass_filter(data['EDA'].values, 1.0, fs, 6)
        
        data = self._detect_arts(data, classifier)
        data = self._correct_eda(data)
        data = frequency_conversion(data, 2)

        for col in ['EDA','filtered_eda','corrected']:
            data[col] = (data[col] - data[col].mean()) / data[col].std()
        return data
    
    def _correct_eda(self, data):
        eda = data['EDA'].values
        k = 0
        level = 7
        olen = len(eda)
        while k*np.power(2, level) < olen:
            k = k + 1
        diff = k*np.power(2, level) - olen
        eda = np.pad(eda, (0, diff), 'constant', constant_values=0)
        type_of_wavelet = 'haar'
        coeff = pywt.swt(eda, type_of_wavelet, level=level, start_level=0, axis=0)
        # od1 = coeff[0][1][0:olen]
        for i in range(0, len(coeff[0][0])):
            for j in range(0, len(coeff)):
                coeff[j][1][i] = 0

        data['corrected'] = pywt.iswt(coeff, type_of_wavelet)[0:olen]
        #eda_data.loc[eda_data[self.classifier[0]] == 1, 'corrected'] = eda_data.loc[eda_data[self.classifier[0]] == 1, 'EDA']
        #eda_data['EDA'] = eda_data['corrected']
        #eda_data['filtered_eda'] = butter_lowpass_filter(eda_data['EDA'].values, 1.0, eda_data.frequency, 6)
        #eda_data = eda_data[['EDA', 'filtered_eda', self.classifier[0]]]
        return data
    
    def _create_feature_df(self,data):
        """
        INPUTS:
            filepath:           string, path to input file
        OUTPUTS:
            features:           DataFrame, index is a list of timestamps for each 5 seconds, contains all the features
            data1:              DataFrame, index is a list of timestamps at 8Hz, columns include AccelZ, AccelY, AccelX,
                                Temp, EDA, filtered_eda
        """
        # Load data1 from q sensor
        wave1sec, wave_half = getWaveletData(data)
        # Compute features for each 5 second epoch
        this_data = data
        this_w1 = wave1sec
        this_w2 = wave_half

        output = getFeatures(this_data, this_w1, this_w2)
        return output


    def _detect_arts(self,eda_data, classifier_list):
        five_sec = 8 * 5
        # Get pickle List and featureNames list
        feature_name_list = [[]] * len(classifier_list)
        for x in range(len(classifier_list)):
            feature_names = getSVMFeatures(classifier_list[x])
            feature_name_list[x] = feature_names
        # Get the number of data1 points, hours, and labels
        rows = len(eda_data)
        num_labels = int(np.ceil(float(rows) / five_sec))
        feature_labels = pd.DataFrame()
        # feature names for DataFrame columns
        all_feature_names = ['raw_amp', 'raw_maxd', 'raw_mind', 'raw_maxabsd', 'raw_avgabsd', 'raw_max2d', 'raw_min2d',
                            'raw_maxabs2d', 'raw_avgabs2d', 'filt_amp', 'filt_maxd', 'filt_mind', 'filt_maxabsd',
                            'filt_avgabsd', 'filt_max2d', 'filt_min2d', 'filt_maxabs2d', 'filt_avgabs2d', 'max_1s_1',
                            'max_1s_2', 'max_1s_3', 'mean_1s_1', 'mean_1s_2', 'mean_1s_3', 'std_1s_1', 'std_1s_2',
                            'std_1s_3', 'median_1s_1', 'median_1s_2', 'median_1s_3', 'aboveZero_1s_1', 'aboveZero_1s_2',
                            'aboveZero_1s_3', 'max_Hs_1', 'max_Hs_2', 'mean_Hs_1', 'mean_Hs_2', 'std_Hs_1', 'std_Hs_2',
                            'median_Hs_1', 'median_Hs_2', 'aboveZero_Hs_1', 'aboveZero_Hs_2']

        features1 = eda_data.groupby(pd.Grouper(freq='5s', origin=eda_data.index[0])).apply(self._create_feature_df)
        features = pd.DataFrame(features1.values.tolist(), columns=all_feature_names, index=features1.index)

        # Al calcular las features, si la ultima ventana de 5s contiene muy pocos datos la función _create_feature_df
        # devuelve nans, dando error en el procesado. En la siguiente linea de codigo procedemos a rellenar los nan
        # con el último valor válido.
        # features = features.fillna(method='ffill')

        for x in range(len(classifier_list)):
            classifier_name = classifier_list[x]
            feature_names = feature_name_list[x]
            feature_labels[classifier_list[x]] = classifyEpochs(features, feature_names, classifier_name)
        feature_labels.index = pd.date_range(eda_data.index[0], periods=num_labels, freq='5s')
        eda_data = eda_data.join(feature_labels, how='outer')#.fillna(method='ffill')

        return eda_data
