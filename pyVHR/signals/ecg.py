import numpy as np
from scipy.signal import find_peaks, stft, lfilter, butter, welch
from plotly.subplots import make_subplots
from plotly.colors import n_colors
import plotly.graph_objects as go
from biosppy.signals import ecg


class ECGsignal:
    """
        Manage (multi-channel, row-wise) BVP signals
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
        Compute the ECG signal by biosppy library
        """
        # TODO: to handle all channels
        data = self.data[0,:]
        out = ecg.ecg(signal=data, sampling_rate=self.fs, show=False)
        self.times = out['heart_rate_ts'] 
        self.bpm = out['heart_rate']
        self.peaksIdX = out['rpeaks']
        
        return self.bpm, self.times

    def autocorr(self):
        from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

        # TODO: to handle all channels
        x = self.data[0,:]
        plot_acf(x)
        plt.show()

        plot_pacf(x)
        plt.show()

    def plot(self):
        """
            Plot the the ECG signals (one channels)
        """
        # TODO: to handle all channels
        data = self.data[0,:]
        N = len(data)
        times = np.linspace(self.startTime, N/self.fs, num=N, endpoint=False)

        # -- plot the channel
        fig = make_subplots(rows=1, cols=1)
        fig.add_trace(go.Scatter(x=times, y=data, name='ECG'), row=1, col=1)
        fig.add_trace(go.Scatter(x=self.peaksIdX/self.fs, y=data[self.peaksIdX], mode='markers', name='peaks'), row=1, col=1)
        fig.update_layout(height=600, width=800)
        fig.show()