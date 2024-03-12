import numpy as np
# import matplotlib.pyplot as plt
# from datetime import timedelta
# from matplotlib.dates import DateFormatter
#from ...features_extractors.cvxEDA import cvxEDA
from ..parsers.eda.art_detect import detect_arts
from .eda_wavelet_thresholding import correct_eda
from ..parsers.utils import butter_lowpass_filter, frequency_conversion
from ..parsers.parameters import *


def __calculate_eda_features(eda_data):

    sampling_interval = 1/eda_data.frequency
    rawdata = np.array(eda_data['EDA'], dtype='float64')
    rawdata = rawdata.flatten()
    r, p, t, l, d, e, o = cvxEDA(rawdata, sampling_interval, options={'reltol': 1e-9, 'show_progress': False})
    eda_data['phasic'] = r
    # EDAdata['tonic'] = EDAdata['filtered'] - EDAdata['phasic']
    eda_data['tonic'] = t
    eda_data['SMNA'] = p

    return eda_data





# if __name__ == '__main__':
#     E4 = 1
#     if E4:
#         from loaders.e4 import load_files
#         directory = 'data/E4'
#         file = 'EDA'
#     else:
#         from loaders.everion import load_files
#         directory = 'data/everion'
#         file = 'gsr'
#     EDAdataALL = load_files(directory, {file: 'EDA'})[0]
#     EDAdata = EDAdataALL[0:14400]  # 1 hour in empatica, 4 hours in everion
#     EDAdata.frequency = EDAdataALL.frequency
#     classifier1 = ['Binary']
#     EDAdata = process_eda(EDAdata, classifier1)
#     fig, axs = plt.subplots(4, 1, figsize=(20, 20))
#     axs[0].plot(EDAdata.index, EDAdata['EDA'].values, label='EDA')
#     axs[1].plot(EDAdata.index, EDAdata['tonic'], label='tonic')
#     axs[2].plot(EDAdata.index, EDAdata['phasic'], label='phasic')
#     axs[3].plot(EDAdata.index, EDAdata['SMNA'], label='SMNA')
#
#     for i in EDAdata[EDAdata[classifier1[0]] != 1].index:
#         axs[0].axvspan(i, i + timedelta(seconds=1/EDAdata.frequency), facecolor='red', alpha=0.7, edgecolor='none')
#         axs[1].axvspan(i, i + timedelta(seconds=1/EDAdata.frequency), facecolor='red', alpha=0.7, edgecolor='none')
#         axs[2].axvspan(i, i + timedelta(seconds=1 / EDAdata.frequency), facecolor='red', alpha=0.7, edgecolor='none')
#         axs[3].axvspan(i, i + timedelta(seconds=1 / EDAdata.frequency), facecolor='red', alpha=0.7, edgecolor='none')
#
#     axs[0].set_xlim([EDAdata.index[0], EDAdata.index[-1]])
#     axs[1].set_xlim([EDAdata.index[0], EDAdata.index[-1]])
#     axs[2].set_xlim([EDAdata.index[0], EDAdata.index[-1]])
#     axs[3].set_xlim([EDAdata.index[0], EDAdata.index[-1]])
#
#     formatter = DateFormatter('%H:%M:%S')
#     plt.gcf().axes[0].xaxis.set_major_formatter(formatter)
#     plt.gcf().axes[1].xaxis.set_major_formatter(formatter)
#     plt.gcf().axes[2].xaxis.set_major_formatter(formatter)
#     plt.gcf().axes[3].xaxis.set_major_formatter(formatter)
#
#     axs[0].legend(loc='upper right')
#     axs[1].legend(loc='upper right')
#     axs[2].legend(loc='upper right')
#     axs[3].legend(loc='upper right')
#     plt.show()
