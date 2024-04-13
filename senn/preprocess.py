"""
This module is used to process the one dimension signal such as spectra.
    Author: Jeill
"""
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.sparse.linalg import spsolve
from sklearn import preprocessing, decomposition, manifold

__all__ = ['gauss_filter_1D',
           'baseline_als',
           'minmax_scale',
           'normalize',
           'clip_data_by_indexes',
           'clip_data_by_shift',
           'PCA',
           'T_SNE']


def gauss_filter_1D(data=None, sigma=1, r=0, step=1, isPadding=False):
    """Gaussian filter for spectra.

        Parameters:
        ----------
            data: {`ndarray`, `pd.DataFrame`}.
                Spectra needing to deal, the data type is a vector of 'ndarray';
            sigma: {int, float}, default=1.
                The value of sigma in gaussian function, which can express different
                space scale in algorithm of SIFT. So this function can be used in SIFT to yield a
                space scale that you want to get.
            r: `int`, default=0.
                Radius of gaussian function, which is used to generate the kernel of gaussian filter.
            step: `int`, default=1.
                Moving step of gaussian kernel;
            isPadding: `bool`, default=`False`.
                It will add 'r' 0 at spectra starting and end respectively if you want the first 'r'
                data and the last 'r' data of the spectra to be filtered when this parameter's value
                is 'True'.

        return:
        ------
            result  :
                Spectra had filtered.
    """
    if data is None:
        raise ValueError('Data is None, please input the data!')

    if r == 0:
        r = int(3 * sigma)

    # gauss filter
    def filter_it(dataInside):
        sizeOfGaussTemp = 2 * r + 1
        gaussTemp = np.array([0] * sizeOfGaussTemp, dtype=np.float64)

        def gaussFun(x):
            return np.exp((-x ** 2) / (2 * (sigma ** 2))) / (sigma * np.sqrt(np.pi * 2))

        for i in range(2 * r + 1):
            gaussTemp[i] = gaussFun(i - r)

        ecg_filtered = dataInside.copy()
        for j in range(r, dataInside.shape[0] - r, step):
            ecg_filtered[j] = dataInside[j - r: j + r + 1].dot(gaussTemp)
        return ecg_filtered

    if isPadding is True:
        padding = np.zeros((r,), dtype=np.float64)
        originSize = data.shape[0]
        data = np.hstack((padding, data, padding))
        result = filter_it(data)[r: originSize + r]
    elif isPadding is False:
        result = filter_it(data)
    else:
        raise ValueError('Please input a bool value! (isPadding == True or False)')
    return result


def baseline_als(data=None, lam=1000, p=0.0002, niter=10):
    """Baseline correction of spectra.

        Parameters:
        ----------
            data : `ndarray`, `pd.DataFrame`
                Spectra needing to correct, the data type is 'ndarray' or 'DataFrame' of pandas.
            lam  : int
            p    : float
            niter: int

        return:
        ------
            result:
                spectra had corrected, the type is same as input.
    """
    if data is None:
        raise ValueError('The data should not be None!')

    def correct_data(y):
        s = len(y)
        # assemble difference matrix
        D0 = sparse.eye(s)
        d1 = [np.ones(s - 1) * -2]
        D1 = sparse.diags(d1, [-1])
        d2 = [np.ones(s - 2) * 1]
        D2 = sparse.diags(d2, [-2])
        D = D0 + D2 + D1
        w = np.ones(s)
        z = None
        for _ in range(niter):
            W = sparse.diags([w], [0])
            Z = W + lam * D.dot(D.transpose())
            

            z = spsolve(Z, w * y)
            w = p * (y > z) + (1 - p) * (y < z)
        return z

    if pd.core.frame.DataFrame == type(data):
        columns_title = data.columns
        values = data.values
    else:
        columns_title = None
        values = data

    if data.shape[1] > 2:
        y_raw = values[:, 1:]
        x = values[:, 0].reshape(values.shape[0], 1)
        y_result = np.ones((y_raw.shape[0], y_raw.shape[1]))
        for i in range(y_raw.shape[1]):
            y_result[:, i] = y_raw[:, i] - correct_data(y_raw[:, i])
        result = np.hstack((x, y_result)).astype(int)
        if pd.core.frame.DataFrame == type(data):
            return pd.DataFrame(result, columns=columns_title)
        else:
            return result
    else:
        values[:, 1] = values[:, 1] - correct_data(values[:, 1])
        if pd.core.frame.DataFrame == type(data):
            return pd.DataFrame(values.astype(int), columns=columns_title)
        else:
            return values.astype(int)


def minmax_scale(spectra_data):
    """Min Max Scale
    """
    spectra_data_values = __get_spectra_data_values(spectra_data)
    raman_shift = spectra_data_values[:, 0]
    intensity = spectra_data_values[:, 1:]
    scaled_data = preprocessing.minmax_scale(intensity)
    return np.hstack((raman_shift, scaled_data)).T


def normalize(spectra_data, p_norm: int = 2):
    """This method is used to normalize the spectra data matrix.

     Spectra_data's first column is 'Raman Shift' and the other columns are 'Intensity' of peaks.

    Parameters:
    ----------
        spectra_data: `ndarray`, `pd.DataFrame`.
            spectra data matrix.
        p_norm: `Int`
            norm which uses to normalize the spectra. If it <= 2, will compute `L1`, `L2`
            norm using `sklearn` lib, otherwise, will compute `L_p` norm using self-definition
            method.
    """
    spectra_data_values = __get_spectra_data_values(spectra_data)

    raman_shift = spectra_data_values[:, 0]
    intensity = spectra_data_values[:, 1:]
    if p_norm == 1:
        normalized_data = preprocessing.normalize(intensity.T, norm='l1')
        return np.vstack((raman_shift.T, normalized_data))
    elif p_norm == 2:
        normalized_data = preprocessing.normalize(intensity.T, norm='l2')
        return np.vstack((raman_shift.T, normalized_data))
    else:
        raise NotImplemented


def clip_data_by_indexes(spectra_data, indexes_range=None):
    """Clipped the spectra data.

    Parameters:
    ----------
    spectra_data: `ndarray`, `pd.DataFrame`.
        A spectra data matrix.
    indexes_range: `tuple`, default=`None`.
        A tuple used to describe clipped range, which the first value and the second value is the
        minimum and maximum index, respectively.
    """
    spectra_data_values = __get_spectra_data_values(spectra_data)
    if indexes_range is None:
        return spectra_data_values
    else:
        return spectra_data_values[:, indexes_range[0]: indexes_range[1] + 1]


def clip_data_by_shift(spectra_data, min_max_range=None):
    """Clip the spectra data.

    Parameters:
    ----------
    spectra_data: `ndarray`, `pd.DataFrame`
        A spectra data matrix.
    min_max_range:`tuple`, default=`None`.
        A tuple used to describe clipped range, which the first value and the second value is the
        minimum and maximum shift, respectively.
    """
    spectra_data_values = __get_spectra_data_values(spectra_data)
    if min_max_range is None:
        return spectra_data_values

    shift = spectra_data_values[0]
    min_shift = min_max_range[0]
    max_shift = min_max_range[1]
    min_index = np.argwhere(shift == min_shift)
    max_index = np.argwhere(shift == max_shift)

    if min_index.shape[0] == 0:
        raise ValueError('Can not find minimum shift {}, please check the shift whether '
                         'it is included in the raman shift.'.format(min_shift))
    if max_index.shape[0] == 0:
        raise ValueError('Can not find maximum shift {}, please check the shift whether '
                         'it is included in the raman shift.'.format(max_shift))

    return clip_data_by_indexes(spectra_data_values, (min_index[0, 0], max_index[0, 0]))


def PCA(spectra_data, n_components=1):
    """Principal component analysis (PCA).

    Parameters:
    ----------
    spectra_data: `ndarray` or `pd.DataFrame`.
        An spectrum data matrix.
    """
    spectra_data_values = __get_spectra_data_values(spectra_data).T

    pca = decomposition.PCA(n_components=n_components)
    return pca.fit_transform(spectra_data_values)


def T_SNE(spectra_data, n_components=1):
    """t-Distributed Stochastic Neighbor Embedding (t-SNE).

    Parameters:
    ----------
    spectra_data: `ndarray` or `pd.DataFrame`.
        An spectrum data matrix.
    """
    spectra_data_values = __get_spectra_data_values(spectra_data).T

    t_sne = manifold.TSNE(n_components=n_components)
    return t_sne.fit_transform(spectra_data_values)


def __get_spectra_data_values(spectra_data):
    if pd.core.frame.DataFrame == type(spectra_data):
        return spectra_data.values
    else:
        return spectra_data
