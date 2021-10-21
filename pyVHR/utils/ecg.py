import numpy as np
from scipy.signal import find_peaks, stft, lfilter, butter, welch
from plotly.subplots import make_subplots
from plotly.colors import n_colors
import plotly.graph_objects as go
from biosppy.signals import ecg


class ECGsignal:
    """
        Manage ECG signals
    """
    verb = False   # verbose (True)
    nFFT = 4*4096  # freq. resolution for STFTs
    step = 1       # step in seconds
    minHz = .75    # 39 BPM - min freq.
    maxHz = 4.    # 240 BPM - max freq.

    def __init__(self, data, fs, startTime=0):
        #self.data = data
        if len(data.shape) == 1:
            self.data = data.reshape(1,-1) # 2D array raw-wise
        self.fs = fs                       # sample rate
        self.startTime = startTime

    def getBPM(self, winsize=5):
        """
        Compute the BPMs (ECG peaks) by biosppy library

        Returns:
            This method returns two variables: a list of BPMs and a list of times (each one correspond to the i-th BPM).
        """
        # TODO: to handle all channels
        data = self.data[0,:]
        out = ecg.ecg(signal=data, sampling_rate=self.fs, show=False)
        self.times = out['heart_rate_ts'] 
        self.bpm = out['heart_rate']
        self.peaksIdX = out['rpeaks']
        
        return self.bpm, self.times
