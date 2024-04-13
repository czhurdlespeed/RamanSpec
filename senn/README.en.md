# SENN

-------

## Description

This project associates with "Deep learning-based spectral extraction for improving the performance of SERS analysis on multiplexed identification and quantitation". In this work, although spectral extraction neural network (SENN) was used to extract pure spectra from trinary mixed solution, it has potential to extract pure component from any mixed signal.

## Abstract

The abstract of our article wille be update when our work is published.

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
