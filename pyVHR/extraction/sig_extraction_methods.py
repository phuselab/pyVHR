from numba import njit, prange, float32
import math
import time
import numpy as np
import PIL.Image
import torchvision.transforms as transforms
from numba import prange, njit
import os
import matplotlib.pyplot as plt


"""
This module defines classes or methods used for signal extraction.
"""

class SignalProcessingParams():
    """
        This class contains usefull parameters used by this module.

        RGB_LOW_TH (numpy.int32): RGB low-threshold value.

        RGB_HIGH_TH (numpy.int32): RGB high-threshold value.
    """
    RGB_LOW_TH = np.int32(55)
    RGB_HIGH_TH = np.int32(200)


@njit(['float32[:,:](uint8[:,:,:], int32, int32)', ], parallel=True, fastmath=True, nogil=True)
def holistic_mean(im, RGB_LOW_TH, RGB_HIGH_TH):
    """
    This method computes the RGB-Mean Signal excluding 'im' pixels
    that are outside the RGB range [RGB_LOW_TH, RGB_HIGH_TH] (extremes are included).

    Args: 
        im (uint8 ndarray): ndarray with shape [rows, columns, rgb_channels].
        RGB_LOW_TH (numpy.int32): RGB low threshold value.
        RGB_HIGH_TH (numpy.int32): RGB high threshold value.
    
    Returns:
        RGB-Mean Signal as float32 ndarray with shape [1,3], where 1 is the single estimator,
        and 3 are r-mean, g-mean and b-mean.
    """
    mean = np.zeros((1, 3), dtype=np.float32)
    mean_r = np.float32(0.0)
    mean_g = np.float32(0.0)
    mean_b = np.float32(0.0)
    num_elems = np.float32(0.0)
    for x in prange(im.shape[0]):
        for y in prange(im.shape[1]):
            if not((im[x, y, 0] <= RGB_LOW_TH and im[x, y, 1] <= RGB_LOW_TH and im[x, y, 2] <= RGB_LOW_TH)
                    or (im[x, y, 0] >= RGB_HIGH_TH and im[x, y, 1] >= RGB_HIGH_TH and im[x, y, 2] >= RGB_HIGH_TH)):
                mean_r += im[x, y, 0]
                mean_g += im[x, y, 1]
                mean_b += im[x, y, 2]
                num_elems += 1.0
    if num_elems > 1.0:
        mean[0, 0] = mean_r / num_elems
        mean[0, 1] = mean_g / num_elems
        mean[0, 2] = mean_b / num_elems
    else:
        mean[0, 0] = mean_r
        mean[0, 1] = mean_g
        mean[0, 2] = mean_b 
    return mean


@njit(['float32[:,:](float32[:,:],uint8[:,:,:],float32, int32, int32)', ], parallel=True, fastmath=True, nogil=True)
def landmarks_mean(ldmks, im, square, RGB_LOW_TH, RGB_HIGH_TH):
    """
    This method computes the RGB-Mean Signal excluding 'im' pixels
    that are outside the RGB range [RGB_LOW_TH, RGB_HIGH_TH] (extremes are included).

    Args: 
        ldmks (float32 ndarray): landmakrs as ndarray with shape [num_landmarks, 5],
             where the second dimension contains y-coord, x-coord, r-mean (value is not important), g-mean (value is not important), b-mean (value is not important).
        im (uint8 ndarray): ndarray with shape [rows, columns, rgb_channels].
        square (numpy.float32): side size of square patches.
        RGB_LOW_TH (numpy.int32): RGB low threshold value.
        RGB_HIGH_TH (numpy.int32): RGB high threshold value.
    
    Returns:
        RGB-Mean Signal as float32 ndarray with shape [num_landmarks, 5], where the second dimension contains y-coord, x-coord, r-mean, g-mean, b-mean.
    """
    r_ldmks = ldmks.astype(np.float32)
    width = im.shape[1]
    height = im.shape[0]
    S = math.floor(square/2)
    lds_mean = np.zeros((ldmks.shape[0], 3), dtype=np.float32)
    num_elems = np.zeros((ldmks.shape[0], ), dtype=np.float32)
    for ld_id in prange(0, r_ldmks.shape[0]):
        if r_ldmks[ld_id, 0] >= 0.0:
            for x in prange(int(r_ldmks[ld_id, 0] - S), int(r_ldmks[ld_id, 0] + S + 1)):
                for y in prange(int(r_ldmks[ld_id, 1] - S), int(r_ldmks[ld_id, 1] + S + 1)):
                    if x >= 0 and x < height and y >= 0 and y < width:
                        if not((im[x, y, 0] <= RGB_LOW_TH and im[x, y, 1] <= RGB_LOW_TH and im[x, y, 2] <= RGB_LOW_TH) or
                               (im[x, y, 0] >= RGB_HIGH_TH and im[x, y, 1] >= RGB_HIGH_TH and im[x, y, 2] >= RGB_HIGH_TH)):
                            lds_mean[ld_id, 0] += np.float32(im[x, y, 0])
                            lds_mean[ld_id, 1] += np.float32(im[x, y, 1])
                            lds_mean[ld_id, 2] += np.float32(im[x, y, 2])
                            num_elems[ld_id] += 1.0
            if num_elems[ld_id] > 1.0:
                r_ldmks[ld_id, 2] = lds_mean[ld_id, 0] / num_elems[ld_id]
                r_ldmks[ld_id, 3] = lds_mean[ld_id, 1] / num_elems[ld_id]
                r_ldmks[ld_id, 4] = lds_mean[ld_id, 2] / num_elems[ld_id]
    return r_ldmks


@njit(['float32[:,:](float32[:,:],uint8[:,:,:],float32, int32, int32)', ], parallel=True, fastmath=True, nogil=True)
def landmarks_median(ldmks, im, square, RGB_LOW_TH, RGB_HIGH_TH):
    """
    This method computes the RGB-Median Signal excluding 'im' pixels
    that are outside the RGB range [RGB_LOW_TH, RGB_HIGH_TH] (extremes are included).

    Args: 
        ldmks (float32 ndarray): landmakrs as ndarray with shape [num_landmarks, 5],
             where the second dimension contains y-coord, x-coord, r-mean (value is not important), g-mean (value is not important), b-mean (value is not important).
        im (uint8 ndarray): ndarray with shape [rows, columns, rgb_channels].
        square (numpy.float32): side size of square patches.
        RGB_LOW_TH (numpy.int32): RGB low threshold value.
        RGB_HIGH_TH (numpy.int32): RGB high threshold value.
    
    Returns:
        RGB-Median Signal as float32 ndarray with shape [num_landmarks, 5], where the second dimension contains y-coord, x-coord, r-mean, g-mean, b-mean.
    """
    r_ldmks = ldmks.astype(np.float32)
    width = float32(im.shape[1])
    height = float32(im.shape[0])
    S = math.floor(square/2.0)
    for ld_id in prange(r_ldmks.shape[0]):
        if r_ldmks[ld_id, 0] >= 0.0:
            x_s = r_ldmks[ld_id, 0] - S
            if x_s < 0.0:
                x_s = 0
            x_e = r_ldmks[ld_id, 0] + S
            if x_e >= height:
                x_e = height
            y_s = r_ldmks[ld_id, 1] - S
            if y_s < 0.0:
                y_s = 0
            y_e = r_ldmks[ld_id, 1] + S
            if y_e >= width:
                y_e = width
            ar = np.copy(im[int(x_s):int(x_e), int(y_s):int(y_e), :])
            f_ar = ar.flatten()
            r = ar[:, :, 0].flatten()
            g = ar[:, :, 1].flatten()
            b = ar[:, :, 2].flatten()
            goodidx = np.ones((ar.shape[0]*ar.shape[1],), dtype=np.int32)
            targets = np.arange(3, f_ar.shape[0], 3)
            for idx in prange(targets.shape[0]):
                i = targets[idx]
                if ((f_ar[i-2] <= RGB_LOW_TH and f_ar[i-1] <= RGB_LOW_TH and f_ar[i] <= RGB_LOW_TH) or
                        (f_ar[i-2] >= RGB_HIGH_TH and f_ar[i-1] >= RGB_HIGH_TH and f_ar[i] >= RGB_HIGH_TH)):
                    goodidx[i % 3] = 0
            goodidx = np.argwhere(goodidx).flatten()
            if goodidx.size < 1 or r.size < 1:
                r_ldmks[ld_id, 2] = np.float32(0.0)
                r_ldmks[ld_id, 3] = np.float32(0.0)
                r_ldmks[ld_id, 4] = np.float32(0.0)
            else:
                r_ldmks[ld_id, 2] = np.float32(np.median(r[goodidx]))
                r_ldmks[ld_id, 3] = np.float32(np.median(g[goodidx]))
                r_ldmks[ld_id, 4] = np.float32(np.median(b[goodidx]))
    return r_ldmks


@njit(['float32[:,:](float32[:,:],uint8[:,:,:],float32[:,:], int32, int32)', ], parallel=True, fastmath=True, nogil=True)
def landmarks_mean_custom_rect(ldmks, im, rects, RGB_LOW_TH, RGB_HIGH_TH):
    """
    This method computes the RGB-Mean Signal excluding 'im' pixels
    that are outside the RGB range [RGB_LOW_TH, RGB_HIGH_TH] (extremes are included).

    Args: 
        ldmks (float32 ndarray): landmakrs as ndarray with shape [num_landmarks, 5],
             where the second dimension contains y-coord, x-coord, r-mean (value is not important), g-mean (value is not important), b-mean (value is not important).
        im (uint8 ndarray): ndarray with shape [rows, columns, rgb_channels].
        rects (float32 ndarray): positive float32 np.ndarray of shape [num_landmarks, 2]. If the list of used landmarks is [1,2,3] 
            and rects_dim is [[10,20],[12,13],[40,40]] then the landmark number 2 will have a rectangular patch of xy-dimension 12x13.
        RGB_LOW_TH (numpy.int32): RGB low threshold value.
        RGB_HIGH_TH (numpy.int32): RGB high threshold value.
    
    Returns:
        RGB-Mean Signal as float32 ndarray with shape [num_landmarks, 5], where the second dimension contains y-coord, x-coord, r-mean, g-mean, b-mean.
    """
    r_ldmks = ldmks.astype(np.float32)
    width = im.shape[1]
    height = im.shape[0]
    lds_mean = np.zeros((ldmks.shape[0], 3), dtype=np.float32)
    num_elems = np.zeros((ldmks.shape[0], ), dtype=np.float32)
    for ld_id in prange(0, r_ldmks.shape[0]):
        if r_ldmks[ld_id, 0] >= 0:
            Sx = math.floor(rects[ld_id, 1]/2)
            Sy = math.floor(rects[ld_id, 0]/2)
            for x in prange(r_ldmks[ld_id, 0] - Sx, r_ldmks[ld_id, 0] + Sx + 1):
                for y in prange(r_ldmks[ld_id, 1] - Sy, r_ldmks[ld_id, 1] + Sy + 1):
                    if x >= 0 and x < height and y >= 0 and y < width:
                        if not((im[x, y, 0] <= RGB_LOW_TH and im[x, y, 1] <= RGB_LOW_TH and im[x, y, 2] <= RGB_LOW_TH) or
                               (im[x, y, 0] >= RGB_HIGH_TH and im[x, y, 1] >= RGB_HIGH_TH and im[x, y, 2] >= RGB_HIGH_TH)):
                            lds_mean[ld_id, 0] += im[x, y, 0]
                            lds_mean[ld_id, 1] += im[x, y, 1]
                            lds_mean[ld_id, 2] += im[x, y, 2]
                            num_elems[ld_id] += 1.0
            if num_elems[ld_id] > 1.0:
                r_ldmks[ld_id, 2] = lds_mean[ld_id, 0] / num_elems[ld_id]
                r_ldmks[ld_id, 3] = lds_mean[ld_id, 1] / num_elems[ld_id]
                r_ldmks[ld_id, 4] = lds_mean[ld_id, 2] / num_elems[ld_id]
    return r_ldmks


@njit(['float32[:,:](float32[:,:],uint8[:,:,:],float32[:,:], int32, int32)', ], parallel=True, fastmath=True, nogil=True)
def landmarks_median_custom_rect(ldmks, im, rects, RGB_LOW_TH, RGB_HIGH_TH):
    """
    This method computes the RGB-Median Signal excluding 'im' pixels
    that are outside the RGB range [RGB_LOW_TH, RGB_HIGH_TH] (extremes are included).

    Args: 
        ldmks (float32 ndarray): landmakrs as ndarray with shape [num_landmarks, 5],
             where the second dimension contains y-coord, x-coord, r-mean (value is not important), g-mean (value is not important), b-mean (value is not important).
        im (uint8 ndarray): ndarray with shape [rows, columns, rgb_channels].
        rects (float32 ndarray): positive float32 np.ndarray of shape [num_landmarks, 2]. If the list of used landmarks is [1,2,3] 
            and rects_dim is [[10,20],[12,13],[40,40]] then the landmark number 2 will have a rectangular patch of xy-dimension 12x13.
        RGB_LOW_TH (numpy.int32): RGB low threshold value.
        RGB_HIGH_TH (numpy.int32): RGB high threshold value.
    
    Returns:
        RGB-Median Signal as float32 ndarray with shape [num_landmarks, 5], where the second dimension contains y-coord, x-coord, r-mean, g-mean, b-mean.
    """
    r_ldmks = ldmks.astype(np.float32)
    width = float32(im.shape[1])
    height = float32(im.shape[0])
    for ld_id in prange(ldmks.shape[0]):
        if r_ldmks[ld_id, 0] >= 0.0:
            Sx = math.floor(rects[ld_id, 1]/2)
            Sy = math.floor(rects[ld_id, 0]/2)
            x_s = r_ldmks[ld_id, 0] - Sx
            if x_s < 0.0:
                x_s = 0
            x_e = r_ldmks[ld_id, 0] + Sx
            if x_e >= height:
                x_e = height
            y_s = r_ldmks[ld_id, 1] - Sy
            if y_s < 0.0:
                y_s = 0
            y_e = r_ldmks[ld_id, 1] + Sy
            if y_e >= width:
                y_e = width
            ar = np.copy(im[int(x_s):int(x_e), int(y_s):int(y_e), :])
            f_ar = ar.flatten()
            r = ar[:, :, 0].flatten()
            g = ar[:, :, 1].flatten()
            b = ar[:, :, 2].flatten()
            goodidx = np.ones((ar.shape[0]*ar.shape[1],), dtype=np.int32)
            targets = np.arange(3, f_ar.shape[0], 3)
            for idx in prange(targets.shape[0]):
                i = targets[idx]
                if ((f_ar[i-2] <= RGB_LOW_TH and f_ar[i-1] <= RGB_LOW_TH and f_ar[i] <= RGB_LOW_TH) or
                        (f_ar[i-2] >= RGB_HIGH_TH and f_ar[i-1] >= RGB_HIGH_TH and f_ar[i] >= RGB_HIGH_TH)):
                    goodidx[i % 3] = 0
            goodidx = np.argwhere(goodidx).flatten()
            if goodidx.size < 1 or r.size < 1:
                r_ldmks[ld_id, 2] = np.int32(0)
                r_ldmks[ld_id, 3] = np.int32(0)
                r_ldmks[ld_id, 4] = np.int32(0)
            else:
                r_ldmks[ld_id, 2] = np.int32(np.median(r[goodidx]))
                r_ldmks[ld_id, 3] = np.int32(np.median(g[goodidx]))
                r_ldmks[ld_id, 4] = np.int32(np.median(b[goodidx]))
    return r_ldmks
