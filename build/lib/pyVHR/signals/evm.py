import argparse
import numpy as np
import cv2
import scipy.signal as signal
import scipy.fftpack as fftpack


def build_gaussian_pyramid(src, levels=3):
    """
    Function: build_gaussian_pyramid
    --------------------------------
        Builds a gaussian pyramid

    Args:
    -----
        src: the input image
        levels: the number levels in the gaussian pyramid

    Returns:
    --------
        A gaussian pyramid
    """
    s = src.copy()
    pyramid = [s]
    print(s.shape)
    for i in range(levels):
        s = cv2.pyrDown(s)
        pyramid.append(s)
        
        print(s.shape)
        
    return pyramid


def gaussian_video(video, levels=3):
    """
    Function: gaussian_video
    ------------------------
       generates a gaussian pyramid for each frame in a video

    Args:
    -----
        video: the input video array
        levels: the number of levels in the gaussian pyramid

    Returns:
    --------
        the gaussian video
    """
    n = video.shape[0]
    for i in range(0, n):
        pyr = build_gaussian_pyramid(video[i], levels=levels)
        gaussian_frame=pyr[-1]
        if i==0:
            vid_data = np.zeros((n, *gaussian_frame.shape))
        vid_data[i] = gaussian_frame
    return vid_data

def temporal_ideal_filter(arr, low, high, fps, axis=0):
    """
    Function: temporal_ideal_filter
    -------------------------------
       Applies a temporal ideal filter to a numpy array
    Args:
    -----
        arr: a numpy array with shape (N, H, W, C)
            N: number of frames
            H: height
            W: width
            C: channels
        low: the low frequency bound
        high: the high frequency bound
        fps: the video frame rate
        axis: the axis of video, should always be 0
    Returns:
    --------
        the array with the filter applied
    """
    fft = fftpack.fft(arr, axis=axis)
    frequencies = fftpack.fftfreq(arr.shape[0], d=1.0 / fps)
    bound_low = (np.abs(frequencies - low)).argmin()
    bound_high = (np.abs(frequencies - high)).argmin()
    fft[:bound_low] = 0
    fft[bound_high:-bound_high] = 0
    fft[-bound_low:] = 0
    iff=fftpack.ifft(fft, axis=axis)
    return np.abs(iff)


def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    """
    Function: butter_bandpass_filter
    --------------------------------
        applies a buttersworth bandpass filter
    Args:
    -----
        data: the input data
        lowcut: the low cut value
        highcut: the high cut value
        fs: the frame rate in frames per second
        order: the order for butter
    Returns:
    --------
        the result of the buttersworth bandpass filter
    """
    omega = 0.5 * fs
    low = lowcut / omega
    high = highcut / omega
    b, a = signal.butter(order, [low, high], btype='band')
    y = signal.lfilter(b, a, data, axis=0)
    return y

def reconstruct_video_g(amp_video, original_video, levels=3):
    """
    Function: reconstruct_video_g
    -----------------------------
        reconstructs a video from a gaussian pyramid and the original

    Args:
    -----
        amp_video: the amplified gaussian video
        original_video: the original video
        levels: the levels in the gaussian video

    Returns:
    --------
        the reconstructed video
    """
    
    print(original_video.shape)
    final_video = np.zeros(original_video.shape)
    for i in range(0, amp_video.shape[0]):
        img = amp_video[i]
        print(img.shape)
        for x in range(levels):
            img = cv2.pyrUp(img)
        
            print(img.shape)
        
        img = img + original_video[i]
        final_video[i] = img
    return final_video