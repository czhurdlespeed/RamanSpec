import os
import torch
import torch.nn as nn
import torch.optim as opt
import numpy as np
import pandas as pd
from  configparser import ConfigParser
import model
import fit_and_valid as f
import decorators
from scipy import signal
import preprocess
from sklearn import preprocessing


# set the random seed
np.random.seed(100)
torch.manual_seed(100)

MODEL_PATH = ''
DATA_PATH = ''
FILE_NAME = ''
DDM = []
MVD = []
TMTD = []

senn = torch.load(os.path.join(MODEL_PATH, 'senn.pt'))

class NewSet(torch.utils.data.Dataset):
    def __init__(self, data):
        self.spectra = data
    
    def __getitem__(self, idx):
        return self.spectra[idx]
    
    def __len__(self):
        return len(self.spectra)

@decorators.RandomSplit(isRandomSplit=False, batchSize=1)
def loadData(filename):
    data = pd.read_csv(os.path.join(DATA_PATH, filename), delimiter='\t')
    data = preprocess.baseline_als(data)
    data = preprocess.clip_data_by_shift(data.T, (400, 1750))
    print(data.shape)
    value = preprocessing.minmax_scale(signal.savgol_filter(data[1:], 11, 3), axis=1)
    return NewSet(value)

def main():
    dataloader = loadData(FILE_NAME)
    with torch.no_grad():
        for i, spectrum in enumerate(dataloader):
            mixture = spectrum.unsqueeze(1).to(torch.float32)
            ddm, mvd, tmtd = senn(mixture)
            DDM.append(ddm.squeeze(1).detach().numpy())
            MVD.append(mvd.squeeze(1).detach().numpy())
            TMTD.append(tmtd.squeeze(1).detach().numpy())
    
    return np.vstack(DDM), np.vstack(MVD), np.vstack(TMTD)

if __name__ == '__main__':
    ddm_o, mvd_o, tmtd_o = main()
    np.savetxt(os.path.join(DATA_PATH, FILE_NAME.replace('.txt', '-ddm_o.csv')), ddm_o.T, delimiter=',')
    np.savetxt(os.path.join(DATA_PATH, FILE_NAME.replace('.txt', '-mvd_o.csv')), mvd_o.T, delimiter=',')
