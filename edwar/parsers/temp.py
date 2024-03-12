from parser import Parser
import numpy as np 


class TEMPparser(Parser):
    def __init__(self):
        super().__init__(inputs={'TEMP': 'ºC'}, outputs={'TEMP': 'ºC'})

    def run(self, data):
        """ Method to run the parser. It checks the input, filters the data and returns the output.

        Args:
            data (pd.DataFrame): Data frame with the temperature data

        Returns:
            pd.DataFrame: Data frame with the filtered temperature data
        """

        self.check_input(self.__class__.__name__, data, self.inputs, self.optional_inputs)
        out = self.filter(data)
        return out

    def filter(self, data):
        """ Method to filters the temperature data based on natural outliers.
        Any temperature value lower than 31ºC is considered an outlier and is interpolated with the closest valid values.

        Args:
            data (pd.DataFrame): Data frame with the temperature data

        Returns:
            pd.DataFrame: Data frame with the filtered temperature data
        """
        if (data['TEMP'] > 31).all():
            return data 
        
        data.loc[data['TEMP'] <= 31] = np.nan
        data['TEMP'] = data['TEMP'].interpolate()

        return data