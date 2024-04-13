import os
import torch
import numpy as np
from torch.optim import Optimizer
from torch.nn import Module, MSELoss
import torch.utils.data as data_util
import matplotlib.pyplot as plt
from decorators import RandomSplit
import dataset

__all__ = ['fit_SENN',
           'test_SENN',
           'get_SENN_output',
           'visualizeResult']

@RandomSplit()
def loadData(dataPath: str):
    files = os.listdir(dataPath)
    totalDataset = []
    for file in files:
        fullPath = os.path.join(dataPath, file)
        data = dataset.SpectraDataset(fullPath, )
        totalDataset.append(data)
    return data_util.ConcatDataset(totalDataset)

def fit_SENN(model, dataloader, criterion, optimizers, comment, epochs=3):
    PRINT_TEMPLATE = '\tEpoch {}/{}: training loss: {:.4f}/{:.4f}/{:.4f} (DDM/MVD/TMTD); ' + \
                        'train similarity: {:.4f}/{:.4f}/{:.4f} (DDM/MVD/TMTD)'
    train_ddm_loss_record = np.zeros(epochs, dtype=np.float64)
    train_ddm_similarity_record = np.zeros(epochs, dtype=np.float64)
    train_mvd_loss_record = np.zeros(epochs, dtype=np.float64)
    train_mvd_similarity_record = np.zeros(epochs, dtype=np.float64)
    train_tmtd_loss_record = np.zeros(epochs, dtype=np.float64)
    train_tmtd_similarity_record = np.zeros(epochs, dtype=np.float64)
    
    for epoch in range(epochs):
        ddm_train_loss = 0
        mvd_train_loss = 0
        tmtd_train_loss = 0
        ddm_train_similarity = 0
        mvd_train_similarity = 0
        tmtd_train_similarity = 0
        train_set_length = 0
        for i_m in range(len(model)):
            model[i_m].train()
        for i, (data, _) in enumerate(dataloader):
            mixture = data[0].unsqueeze(1).to(torch.float32)
            ddm = data[1].to(torch.float32)
            mvd = data[2].to(torch.float32)
            tmtd = data[3].to(torch.float32)
            
            for i_o in range(len(optimizers)):
                optimizers[i_o].zero_grad()
            
            predict_ddm = model[1](model[0](mixture))
            predict_mvd = model[2](model[0](mixture))
            predict_tmtd = model[3](model[0](mixture))
            ddm_loss = criterion[0](predict_ddm.squeeze(1), ddm)
            mvd_loss = criterion[1](predict_mvd.squeeze(1), mvd)
            tmtd_loss = criterion[2](predict_tmtd.squeeze(1), tmtd)
            ddm_train_loss += ddm_loss
            mvd_train_loss += mvd_loss
            tmtd_train_loss += tmtd_loss
            ddm_similarity = comment(predict_ddm.squeeze(1), ddm)
            mvd_similarity = comment(predict_mvd.squeeze(1), mvd)
            tmtd_similarity = comment(predict_tmtd.squeeze(1), tmtd)
            ddm_train_similarity += abs(sum(ddm_similarity.detach().numpy()))
            mvd_train_similarity += abs(sum(mvd_similarity.detach().numpy()))
            tmtd_train_similarity += abs(sum(tmtd_similarity.detach().numpy()))
            train_set_length += dataloader.batch_size
            ddm_loss.backward()
            mvd_loss.backward()
            tmtd_loss.backward()
            for i_opt in range(3):
                optimizers[i_opt].step()
        ddm_average_simi_train = ddm_train_similarity / train_set_length
        mvd_average_simi_train = mvd_train_similarity / train_set_length
        tmtd_average_simi_train = tmtd_train_similarity / train_set_length
        train_ddm_loss_record[epoch] = ddm_train_loss
        train_mvd_loss_record[epoch] = mvd_train_loss
        train_tmtd_loss_record[epoch] = tmtd_train_loss
        train_ddm_similarity_record[epoch] = ddm_average_simi_train
        train_mvd_similarity_record[epoch] = mvd_average_simi_train
        train_tmtd_similarity_record[epoch] = tmtd_average_simi_train
        print(PRINT_TEMPLATE.format(epoch + 1, epochs, ddm_train_loss, mvd_train_loss, tmtd_train_loss,
                                    ddm_average_simi_train, mvd_average_simi_train, tmtd_average_simi_train))
    return {'model': model,
            'ddm_loss': train_ddm_loss_record,
            'mvd_loss': train_mvd_loss_record,
            'tmtd_loss': train_tmtd_loss_record,
            'ddm_simi': train_ddm_similarity_record,
            'mvd_simi': train_mvd_similarity_record,
            'tmtd_simi': train_tmtd_similarity_record}


def test_SENN(model, dataloader, criterions, comment):
    PRINT_TEMPLATE = '\ttest loss: {:.4f}/{:.4f}/{:.4f} (ddm/mvd/tmtd); ' +\
                     'test similarity: {:.4f}/{:.4f}/{:.4f} (ddm/mvd/tmtd)'
    ddm_test_loss = 0
    mvd_test_loss = 0
    tmtd_test_loss = 0
    ddm_test_similarity = 0
    mvd_test_similarity = 0
    tmtd_test_similarity = 0
    test_set_length = 0
    with torch.no_grad():
        for i, (data, _) in enumerate(dataloader):
            mixture = data[0].unsqueeze(1).to(torch.float32)
            ddm = data[1].to(torch.float32)
            mvd = data[2].to(torch.float32)
            tmtd = data[3].to(torch.float32)

            pred_ddm = model[1](model[0](mixture))
            pred_mvd = model[2](model[0](mixture))
            pred_tmtd = model[3](model[0](mixture))
            ddm_loss = criterions[0](pred_ddm.squeeze(1), ddm)
            mvd_loss = criterions[1](pred_mvd.squeeze(1), mvd)
            tmtd_loss = criterions[2](pred_tmtd.squeeze(1), tmtd)
            ddm_test_loss += ddm_loss
            mvd_test_loss += mvd_loss
            tmtd_test_loss += tmtd_loss
            ddm_similarity = comment(pred_ddm.squeeze(1), ddm)
            mvd_similarity = comment(pred_mvd.squeeze(1), mvd)
            tmtd_similarity = comment(pred_tmtd.squeeze(1), tmtd)
            ddm_test_similarity += abs(sum(ddm_similarity.detach().numpy()))
            mvd_test_similarity += abs(sum(mvd_similarity.detach().numpy()))
            tmtd_test_similarity += abs(sum(tmtd_similarity.detach().numpy()))
            test_set_length += dataloader.batch_size
        ddm_average_simi_test = ddm_test_similarity / test_set_length
        mvd_average_simi_test = mvd_test_similarity / test_set_length
        tmtd_average_simi_test = tmtd_test_similarity / test_set_length
        print(PRINT_TEMPLATE.format(ddm_test_loss, mvd_test_loss, tmtd_test_loss,
                                    ddm_average_simi_test, mvd_average_simi_test, tmtd_average_simi_test))
        return {'ddm_simi': ddm_average_simi_test,
                'mvd_simi': mvd_average_simi_test,
                'tmtd_simi': tmtd_average_simi_test,
                'ddm_loss': ddm_test_loss,
                'mvd_loss': mvd_test_loss,
                'tmtd_loss': tmtd_test_loss}

def get_SENN_output(model: Module, loader, num: int = None):
    model.eval()
    pure_spectra_real_record = {'DDM0.01': [], 'DDM0.10': [], 'DDM1.00': [], 
                                'MVD0.01': [], 'MVD0.10': [], 'MVD1.00': [], 
                                'TMTD0.05': [], 'TMTD0.50': [], 'TMTD1.00': []}
    pure_spectra_pred_record = {'DDM0.01': [], 'DDM0.10': [], 'DDM1.00': [], 
                                'MVD0.01': [], 'MVD0.10': [], 'MVD1.00': [], 
                                'TMTD0.05': [], 'TMTD0.50': [], 'TMTD1.00': []}

    def addToRecord(ddm, mvd, tmtd, ddm_k: str, mvd_k: str, tmtd_k: str, which='pred'):
        if which == 'pred':
            pure_spectra_pred_record[ddm_k].append(ddm)
            pure_spectra_pred_record[mvd_k].append(mvd)
            pure_spectra_pred_record[tmtd_k].append(tmtd)
        else:
            pure_spectra_real_record[ddm_k].append(ddm)
            pure_spectra_real_record[mvd_k].append(mvd)
            pure_spectra_real_record[tmtd_k].append(tmtd)

    with torch.no_grad():
        for i, (data, category) in enumerate(loader):
            mixture = data[0].unsqueeze(1).to(torch.float32)
            ddm = data[1].to(torch.float32)
            mvd = data[2].to(torch.float32)
            tmtd = data[3].to(torch.float32)
            batch_size = mvd.shape[0]
            ddm_o, mvd_o, tmtd_o = model(mixture)
            for j in range(batch_size):
                ddm_k, mvd_k, tmtd_k = parseKeyByCategory(category[j])
                addToRecord(ddm_o.squeeze(1)[j].detach().numpy(),
                            mvd_o.squeeze(1)[j].detach().numpy(),
                            tmtd_o.squeeze(1)[j].detach().numpy(),
                            ddm_k, mvd_k, tmtd_k)
                addToRecord(ddm.detach().numpy()[j], mvd.detach().numpy()[j], 
                            tmtd.detach().numpy()[j], ddm_k, mvd_k, tmtd_k, which='real')
            if num is None or num <= 0:
                pass
            elif i >= (num - 1):
                break
    return {'predict': pure_spectra_pred_record,
            'real': pure_spectra_real_record}

def parseKeyByCategory(category: str):
    s, c = category.split('=')
    substance = s.split('-')
    concentration = c.split('-')
    return [substance[0]+concentration[0],
            substance[1]+concentration[1],
            substance[2]+concentration[2]]

def visualizeResult(ddm: [], mvd:[], tmtd:[]):
    pred_ddm = np.loadtxt(ddm[0], delimiter=',').T
    real_ddm = np.loadtxt(ddm[1], delimiter=',').T
    pred_mvd = np.loadtxt(mvd[0], delimiter=',').T
    real_mvd = np.loadtxt(mvd[1], delimiter=',').T
    pred_tmtd = np.loadtxt(tmtd[0], delimiter=',').T
    real_tmtd = np.loadtxt(tmtd[1], delimiter=',').T

    plt.subplot(311)
    plt.plot(pred_ddm[0], 'r')
    plt.plot(real_ddm[0], 'b')
    plt.title('ddm')
    plt.subplot(312)
    plt.plot(pred_mvd[0], 'r')
    plt.plot(real_mvd[0], 'b')
    plt.title('mvd')
    plt.subplot(313)
    plt.plot(pred_tmtd[0], 'r')
    plt.plot(real_tmtd[0], 'b')
    plt.title('tmtd')
    plt.show()

