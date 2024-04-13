import torch
import pandas as pd
import numpy as np
from scipy import signal
from configparser import ConfigParser
import preprocess
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler

__all__ = ['SpectraDataset']

# configParser = ConfigParser()
# configParser['OUTPUT'] = {}

class SpectraDataset(torch.utils.data.Dataset):
    def __init__(self, file):
        # Skipping the first row, using the second for headers
        data = pd.read_csv(file, header=1)
        
        # Separating Raman shifts and actual data
        self.raman_shifts = data['raman_shifts'].values[:]  # Assuming the first value under 'raman_shifts' is a placeholder or text
       
        self.spectra_data = data.drop(columns=['raman_shifts']).values[:]

        # Transposing to get spectra in rows
        self.spectra_data = self.spectra_data.T

        self.labels = data.columns[1:]  # Assuming the first column is not a label but the Raman shifts
        self.dataGroup, self.shift = self.parseComponents()

    def __getitem__(self, idx):
        # Returns the spectral data along with its label
        return self.dataGroup[idx], self.labels[idx]

    def __len__(self):
        # Length is now based on the number of spectra available
        return len(self.dataGroup)

    def parseComponents(self):
        # Process each spectrum similarly to your original approach
        processed_data = []
        for spectrum in self.spectra_data:
            # Assuming preprocess functions are adapted for 1D arrays
       
            spectrum_processed = preprocess.baseline_als(np.vstack((self.raman_shifts, spectrum)).T).T
            
            spectrum_processed = preprocess.clip_data_by_shift(spectrum_processed, (640, 1600))
            shift = spectrum_processed[0]
            values = spectrum_processed[1]
            
            #print(shift)
            #print(values)
            #print(shift.shape)
            #print(values.shape)
            values = MinMaxScaler().fit_transform(signal.savgol_filter(values.reshape(-1, 1), 11, 3)).flatten()
            processed_data.append(values)

        return processed_data, shift

    @staticmethod
    def __get_min_size__(matrix: list):
        if len(matrix) == 0:
            return 0
        else:
            min_size = len(matrix[0])
            for i in range(len(matrix)):
                temp = len(matrix[i])
                if temp < min_size:
                    min_size = temp
            return min_size
