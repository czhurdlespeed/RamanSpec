import torch
import torch.nn as nn
import torch.optim as opt
import numpy as np
from  configparser import ConfigParser
import model
import fit_and_valid as f

# set the random seed
np.random.seed(100)
torch.manual_seed(100)
# 'config.ini' parser
config = ConfigParser()
config.read('config.ini', encoding='utf-8')
# load dataset
DATA_PATH = './data'
trainDataLoader, testDataLoader = f.loadData(DATA_PATH)
linearLen = config.getint('OUTPUT', 'spectrumlen')
# save path
ROOT_DIR = './warehouse/train-result/'
# model modules
modules = [
    model.PreprocessUnit(),
    model.ExtractorUnit(linearLen),
    model.ExtractorUnit(linearLen),
    model.ExtractorUnit(linearLen)
]
# losses
criterions = [
    nn.MSELoss(),
    nn.MSELoss(),
    nn.MSELoss()
]
# config optimizer and learning rate
optimizers = [
    opt.Adam([{'params': modules[1].parameters()}, {'params':modules[0].parameters()}], 0.001),
    opt.Adam(modules[2].parameters(), 0.001),
    opt.Adam(modules[3].parameters(), 0.001)
]
# evaluate result
comment = nn.CosineSimilarity()
# train the model
print(f"{'Start training model!':=^100s}")
tape = f.fit_SENN(modules, trainDataLoader, criterions, optimizers, comment, epochs=50)
print(f"{'Start training model!':=^100s}")
trained_model = tape['model']
senn = model.SENN(trained_model[0], trained_model[1], trained_model[2], trained_model[3])
torch.save(senn, './warehouse/model/senn.pt')
print(f"{'Start test model!':=^100s}")
f.test_SENN(trained_model, testDataLoader, criterions, comment)
output = f.get_SENN_output(senn, testDataLoader)

for key in tape.keys():
    if key == 'model':
        continue
    np.savetxt(ROOT_DIR + 'train-' + key + '.csv', tape[key].T, delimiter=',')

pred = output['predict']
real = output['real']

for key in pred:
    pred_spectra = np.vstack(pred[key])
    np.savetxt(ROOT_DIR + 'test-pred-output-' + key + '.csv', pred_spectra.T, delimiter=',')

for key in real:
    real_spectra = np.vstack(real[key])
    np.savetxt(ROOT_DIR + 'test-real-output-' + key + '.csv', real_spectra.T, delimiter=',')

f.visualizeResult(
    [ROOT_DIR+'test-pred-output-DDM0.01.csv', ROOT_DIR+'test-real-output-DDM0.01.csv'],
    [ROOT_DIR+'test-pred-output-MVD0.01.csv', ROOT_DIR+'test-real-output-MVD0.01.csv'],
    [ROOT_DIR+'test-pred-output-TMTD0.05.csv', ROOT_DIR+'test-real-output-TMTD0.05.csv']
)
