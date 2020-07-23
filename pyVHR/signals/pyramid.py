import numpy as np
import cv2
#import scipy.fftpack as fftpack
import scipy


def gaussian_video(video, pyramid_levels):
    """Create a gaussian representation of a video"""
    vid_data = None
    for x in range(0, video.shape[0]):
        frame = video[x]
        gauss_copy = np.ndarray(shape=frame.shape, dtype="float")
        gauss_copy[:] = frame
        for i in range(pyramid_levels):
            gauss_copy = gausPyrDown(gauss_copy)
        if x == 0:
            vid_data = np.zeros((video.shape[0], gauss_copy.shape[0], gauss_copy.shape[1], 3))
        vid_data[x] = gauss_copy
    return vid_data

def gausPyrDown(frame,sz=5,sigma=1):
    height, width, channel = frame.shape
    convFrame = np.zeros(shape=(height, width,channel), dtype="float")
    kernel = gausKernel(sz=sz,sigma=sigma)
    for channel_i in range(channel):
        convFrame[:,:,channel_i] = scipy.ndimage.convolve(frame[:,:,channel_i],kernel)
    downFrame = convFrame[::2,::2,:]
    return downFrame

def gausPyrUp(frame,sz=5,sigma=1):
    height, width, channel = frame.shape
    upFrame = np.zeros(shape=(2*height, 2*width,channel), dtype="float")
    kernel = gausKernel(sz=sz,sigma=sigma)
    for channel_i in range(channel):
        upFrame[::2,::2,channel_i] = frame[:,:,channel_i]
        upFrame[:,:,channel_i] = scipy.ndimage.convolve(upFrame[:,:,channel_i],kernel*4)
    return upFrame

# Define the Gaussian kernel
def gausKernel(sz = 5,sigma=1):
    kernel = np.zeros((sz,sz))
    ### STUDENT: Implement the Gaussian kernel
    sz_1 = int(sz/2)
    for x in (np.arange(sz)-sz_1):
        for y in (np.arange(sz)-sz_1):
            ix,iy = x+sz_1,y+sz_1
            kernel[ix,iy] = np.exp(-(x**2+y**2)/(2.0*sigma**2))
    kernel = kernel / np.sum(kernel)
    ### STUDENT END
    return kernel

# We will use your previously implemented temporal bandpass filter, so make sure it works！
def temporal_bandpass_filter(data, fps, freq_min=0.833, freq_max=1, axis=0):
    # Inputs:
    # data: video data of shape #frames x height x width x #channel (3,RGB)
    # fps: frames per second (30)
    # freq_min, freq_max: cut-off frequencies for bandpass filter
    # axis: dimension along which to apply FFT (default:0, 
    #       time domain <->for a single pixel along all frames)
    # Output:
    #        Band-passed video data, with only frequency components (absolute value) 
    #        between freq_min and freq_max preserved
    #        of shape #frames x height x width x #channel (3,RGB)
    data_process = np.zeros(data.shape)
    sample_interval = 1.0/fps
    for x in range(data.shape[1]):
        for y in range(data.shape[2]):
            for z in range(data.shape[3]):
                # the bandpass_filter is YOUR implementation!
                data_process[:,x,y,z] = bandpass_filter(data[:,x,y,z], sample_interval, freq_min, freq_max)
    return data_process



# Implement the temporal bandpass filter
def bandpass_filter(x, sample_interval, freq_min, freq_max):
    # Inputs:
    # x: temporal signal of shape (N,)
    # sample_interval: the temporal sampling interval.
    # freq_min, freq_max: cut-off frequencies for bandpass filter
    # Output:
    # Band-passed signal, with only frequency components (absolute value) 
    #      between freq_min and freq_max  preserved

    ### STUDENT: Implement the bandpass filter. 
    ### Feel free to use numpy.fft.fft, numpy.fft.fftfreq, numpy.fft.ifft
    X = np.fft.fft(x)
    frequencies = np.fft.fftfreq(len(x), d=sample_interval)
    bound_low = (np.abs(frequencies - freq_min)).argmin()
    bound_high = (np.abs(frequencies - freq_max)).argmin()
    X[:bound_low] = 0
    X[bound_high:-bound_high] = 0
    X[-bound_low:] = 0
    band_pass_signal = np.abs(np.fft.ifft(X, axis=0))
    ### STUDENT END
    return band_pass_signal


# Utility function: used to convert numpy array to comform with video format
def convertScaleAbs(frame):
    outFrame = np.ndarray(shape=frame.shape, dtype="uint8")
    for channel_i in range(3):
        outFrame[:,:,channel_i] = np.clip(np.abs(frame[:,:,channel_i]),0,255).astype(np.uint8).copy()
    return outFrame


def combine_pyramid_and_save(g_video, orig_video, enlarge_multiple, fps):
    """Combine a gaussian video representation with the original"""
    
    width, height = orig_video.shape[2], orig_video.shape[1]
    mag_data = np.zeros(orig_video.shape, dtype='uint8')
    for x in range(0, g_video.shape[0]):
        img = np.ndarray(shape=g_video[x].shape, dtype='float')
        img[:] = g_video[x]
        for i in range(enlarge_multiple):
            img = gausPyrUp(img)
        img[:height, :width] = img[:height, :width] + orig_video[x]
        res = convertScaleAbs(img[:height, :width])
        mag_data[x] = res
    return mag_data


