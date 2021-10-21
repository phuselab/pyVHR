import numpy as np
import scipy.sparse
import time
import os
import matplotlib.pyplot as plt
from numba import prange, jit
from scipy import stats
import statistics
from scipy.signal import butter, filtfilt, savgol_filter


"""
This module contains a collection of filter methods.

FILTER METHOD SIGNATURE
A Filter method must accept theese parameters:
    > signal -> RGB signal as float32 ndarray with shape [num_estimators, rgb_channels, num_frames],
                or BVP signal as float32 ndarray with shape [num_estimators, num_frames].
    > **kargs [OPTIONAL] -> usefull parameters passed to the filter method.
It must return a filtered signal with the same shape as the input signal.
"""

def apply_filter(windowed_sig, filter_func, fps=None, params={}):
    """
    Apply a filter method to a windowed RGB signal or BVP signal. 

    Args:
        windowed_sig: list of length num_window of RGB signal as float32 ndarray with shape [num_estimators, rgb_channels, num_frames],
                      or BVP signal as float32 ndarray with shape [num_estimators, num_frames].
        filter_func: filter method that accept a 'windowed_sig' (pyVHR implements some filters in pyVHR.BVP.filters).
        params (dict): usefull parameters passed to the filter method.
    
    Returns:
        A filtered signal with the same shape as the input signal.
    """
    if 'fps' in params and params['fps'] == 'adaptive' and fps is not None:
        params['fps'] = np.float32(fps)
    filtered_windowed_sig = []
    for idx in range(len(windowed_sig)):
        transform = False
        sig = np.copy(windowed_sig[idx])
        if len(sig.shape) == 2:
            transform = True
            sig = np.expand_dims(sig, axis=1)
        if params == {}:
            filt_temp = filter_func(sig)
        else:
            filt_temp = filter_func(sig, **params)
        if transform:
            filt_temp = np.squeeze(filt_temp, axis=1)
        filtered_windowed_sig.append(filt_temp)
    return filtered_windowed_sig


# ------------------------------------------------------------------------------------- #
#                                     FILTER METHODS                                    #
# ------------------------------------------------------------------------------------- #


def BPfilter(sig, **kargs):
    """
    Band Pass filter (using BPM band) for RGB signal and BVP signal.

    The dictionary parameters are: {'minHz':float, 'maxHz':float, 'fps':float, 'order':int}
    """
    x = np.array(np.swapaxes(sig, 1, 2))
    b, a = butter(kargs['order'], Wn=[kargs['minHz'],
                                      kargs['maxHz']], fs=kargs['fps'], btype='bandpass')
    y = filtfilt(b, a, x, axis=1)
    y = np.swapaxes(y, 1, 2)
    return y


def zscore(sig):
    """
    Z-score filter for RGB signal and BVP signal.
    """
    x = np.array(np.swapaxes(sig, 1, 2))
    y = stats.zscore(x, axis=2)
    y = np.swapaxes(y, 1, 2)
    return y

 
def detrend(X, **kargs):
    """
    Detrending filter for RGB signal and BVP signal. 

    This code is based on the following article "An advanced detrending method with application
    to HRV analysis". Tarvainen et al., IEEE Trans on Biomedical Engineering, 2002.

    Code like: https://www.idiap.ch/software/bob/docs/bob/bob.rppg.base/v1.0.3/_modules/bob/rppg/cvpr14/filter_utils.html

    The dictionary parameters are: {'detLambda':int,}. Where 'detLambda' is the smoothing parameter.
    """
    if 'detLambda' not in kargs:
        kargs['detLambda'] = 10

    X = np.swapaxes(X, 1, 2)
    result = np.zeros_like(X)
    i = 0
    t = X.shape[1]
    l = t/kargs['detLambda']  # lambda
    I = np.identity(t)
    D2 = scipy.sparse.diags([1, -2, 1], [0, 1, 2], shape=(t-2, t)).toarray()
    Hinv = np.linalg.inv(I+l**2*(np.transpose(D2).dot(D2)))
    for est in X:
        detrendedX = (I - Hinv).dot(est)
        result[i] = detrendedX
        i += 1
    result = np.swapaxes(result, 1, 2)
    return result.astype(np.float32)

def zeromean(X):
    """
    Zero Mean filter for RGB signal and BVP signal. 
    """
    M = np.mean(X, axis=2)
    return X - np.expand_dims(M, axis=2) 

def sg_detrend(X, **kargs):
    """
    Remove the low-frequency components with the low-pass filter developed by Savitzky-Golay.
    It can be used for RGB signals and BVP signals.

    The dictionary parameters are: {'window_length':int, 'polyorder':int}. Where
    'window_length' is the length of the filter window (i.e. the number of coefficients). `window_length` must be a positive odd integer.
    'polyorder' is the order of the polynomial used to fit the samples. `polyorder` must be less than `window_length`.
    """
    if 'window_length' not in kargs:
        kargs['window_length'] = 31
    if 'polyorder' not in kargs:
        kargs['polyorder'] = 5

    trend = savgol_filter(X, window_length=kargs['window_length'], polyorder=kargs['polyorder'], axis=2)
    return X - trend


@jit(["int32[:](float32[:,:,:], int32, int32)", "int32[:](float64[:,:,:], int32, int32)", "int32[:](int32[:,:,:], int32, int32)"], nopython=True, nogil=True, parallel=True, fastmath=True)
def kernel_rgb_filter_th(sig, RGB_LOW_TH, RGB_HIGH_TH):
    """
    This method performs the Color Threshold filter for RGB signal only.
    Please refer to the method pyVHR.BVP.filters.rgb_filter_th.
    """
    goodidx = np.ones((sig.shape[0],), dtype=np.int32)
    for idx in prange(sig.shape[0]):
        b_in = 1
        for f in prange(sig.shape[2]):
            if ((sig[idx][0][f] <= RGB_LOW_TH and sig[idx][1][f] <= RGB_LOW_TH and sig[idx][2][f] <= RGB_LOW_TH) or
                    (sig[idx][0][f] >= RGB_HIGH_TH and sig[idx][1][f] >= RGB_HIGH_TH and sig[idx][2][f] >= RGB_HIGH_TH)):
                b_in = 0
        goodidx[idx] = b_in
    return goodidx


def rgb_filter_th(sig, **kargs):
    """
    Color Threshold filter for RGB signal only.

    The i_th estimators is filterd if its signal is outside the
    LOW/HIGH thresholds color interval in at least one frame.

    The dictionary parameters are: {'RGB_LOW_TH':int, 'RGB_HIGH_TH':int}.
    Where 'RGB_LOW_TH' and 'RGB_HIGH_TH' are RGB value ([0,255]) which describe the filter thresholds.
    """
    goodidx = kernel_rgb_filter_th(
        sig, np.int32(kargs['RGB_LOW_TH']), np.int32(kargs['RGB_HIGH_TH']))
    if np.sum(goodidx) == 0:
        return np.zeros((0, sig.shape[1], sig.shape[2]))
    return np.copy(sig[np.argwhere(goodidx).flatten()])
