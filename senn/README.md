# SENN

-------

## Description

This project associates with "Deep Learning-Based Spectral Extraction for Improving the Performance of Surface-Enhanced Raman Spectroscopy Analysis on Multiplexed Identification and Quantitation". In this work, although spectral extraction neural network (SENN) was used to extract pure spectra from trinary mixed solution, it has potential to extract pure component from any mixed signal.

## Abstract

Surface-enhanced Raman spectroscopy (SERS) has been recognized as a promising analytical technique for its capability of providing molecular fingerprint information and avoiding interference of water. Nevertheless, direct SERS detection of complicated samples without pretreatment to achieve the high-efficiency identification and quantitation in a multiplexed way is still a challenge. In this study, a novel spectral extraction neural network (SENN) model was proposed for synchronous SERS detection of each component in mixed solutions using a demonstration sample containing diquat dibromide (DDM), methyl viologen dichloride (MVD), and tetramethylthiuram disulfide (TMTD). A SERS spectra dataset including 3600 spectra of DDM, MVD, TMTD, and their mixtures was first constructed to train the SENN model. After the training step, the cosine similarity of the SENN model can achieve 0.999, 0.997, and 0.994 for DDM, MVD, and TMTD, respectively, which means that the spectra extracted from the mixture are highly consistent with those collected by the SERS experiment of the corresponding pure samples. Furthermore, a convolutional neural network model for quantitative analysis is combined with the SENN, which can simultaneously and rapidly realize the qualitative and quantitative SERS analysis of mixture solutions with lower than 8.8% relative standard deviation. The result demonstrates that the proposed strategy has great potential in improving SERS analysis in environmental monitoring, food safety, and so on.

## Major libraries

|  Library  |  Version  |
|:---------:|:---------:|
|PyTorch| 1.7.1 |
|Numpy | 1.18.5 |
|Pandas| 1.1.5 |
|Scipy | 1.4.1 |
|Scikit-learn | 0.23.2 |

## How to use

For most situation, you can implement the extractive work by modifying the `main.py`script to add more or less `Extractor` unit according to the number in the mixed signal. Furthermore, the functions in `fit_and_valid.py` should be rewrited so that the model can be exactly trained.

Additionally, you can directly load `senn.pt` model from `.warehouse/model/` and further retrain the model using customized data set.

### Dataset

Each file in `.data/` includes 400 spectra, the first 100 spectra are mixed spectra which mixture ratio is shown in file name. The others are the pure spectra for DDM, MVD and TMTD recorded from experiments, and each component contains 100 spectra with concentration described in file name.
