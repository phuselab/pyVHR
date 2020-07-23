import numpy as np
from scipy.signal import butter, lfilter, filtfilt, freqz
from scipy import signal

def BPfilter(x, minHz, maxHz, fs, order=6):
    """Band Pass filter (using BPM band)"""

    #nyq = fs * 0.5
    #low = minHz/nyq
    #high = maxHz/nyq

    #print(low, high)
    #-- filter type
    #print('filtro=%f' % minHz)
    b, a = butter(order, Wn=[minHz, maxHz], fs=fs, btype='bandpass')
    #TODO verificare filtfilt o lfilter
    #y = lfilter(b, a, x)
    y = filtfilt(b, a, x)

    #w, h = freqz(b, a)


    #import matplotlib.pyplot as plt
    #fig, ax1 = plt.subplots()
    #ax1.set_title('Digital filter frequency response')
    #ax1.plot((fs * 0.5 / np.pi) * w, abs(h), 'b')
    #ax1.set_ylabel('Amplitude [dB]', color='b')
    #plt.show()
    return y

def zeroMeanSTDnorm(x):
    # -- normalization along rows (1-3 channels)
    mx = x.mean(axis=1).reshape(-1,1)
    sx = x.std(axis=1).reshape(-1,1)
    y = (x - mx) / sx
    return y