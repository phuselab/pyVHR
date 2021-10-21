import cupy
import numpy as np
from scipy.signal import welch
from scipy.signal import find_peaks, stft
from plotly.subplots import make_subplots
from plotly.colors import n_colors
import plotly.graph_objects as go
from scipy.interpolate import interp1d
from pyVHR.plot.visualize import VisualizeParams
from pyVHR.BPM.utils import *

"""
This module contains all the class and the methods for transforming 
a BVP signal in a BPM signal.
"""


class BVPsignal:
    """
    Manage (multi-channel, row-wise) BVP signals, and transforms it in BPM.
    This class is used to obtain BPM signal from ground truth BVP signal.
    """
    nFFT = 2048  # freq. resolution for STFTs
    step = 1       # step in seconds

    def __init__(self, data, fs, startTime=0, minHz=0.75, maxHz=4., verb=False):
        if len(data.shape) == 1:
            self.data = data.reshape(1, -1)  # 2D array raw-wise
        else:
            self.data = data
        self.fs = fs                       # sample rate
        self.startTime = startTime
        self.verb = verb
        self.minHz = minHz
        self.maxHz = maxHz

    def spectrogram(self, winsize=5):
        """
        Compute the BVP signal spectrogram restricted to the
        band 42-240 BPM by using winsize (in sec) samples.
        """

        # -- spect. Z is 3-dim: Z[#chnls, #freqs, #times]
        F, T, Z = stft(self.data,
                       self.fs,
                       nperseg=self.fs*winsize,
                       noverlap=self.fs*(winsize-self.step),
                       boundary='even',
                       nfft=self.nFFT)
        Z = np.squeeze(Z, axis=0)

        # -- freq subband (0.75 Hz - 4.0 Hz)
        minHz = 0.65
        maxHz = 4.0
        band = np.argwhere((F > minHz) & (F < maxHz)).flatten()
        self.spect = np.abs(Z[band, :])     # spectrum magnitude
        self.freqs = 60*F[band]            # spectrum freq in bpm
        self.times = T                     # spectrum times

        # -- BPM estimate by spectrum
        self.bpm = self.freqs[np.argmax(self.spect, axis=0)]

    def displaySpectrum(self, display=False, dims=3):
        """Show the spectrogram of the BVP signal"""

        # -- check if bpm exists
        try:
            bpm = self.bpm
        except AttributeError:
            self.spectrogram()
            bpm = self.bpm

        t = self.times
        f = self.freqs
        S = self.spect

        fig = go.Figure()
        fig.add_trace(go.Heatmap(z=S, x=t, y=f, colorscale="viridis"))
        fig.add_trace(go.Scatter(
            x=t, y=bpm, name='Frequency Domain', line=dict(color='red', width=2)))

        fig.update_layout(autosize=False, height=420, showlegend=True,
                          title='Spectrogram of the BVP signal',
                          xaxis_title='Time (sec)',
                          yaxis_title='BPM (60*Hz)',
                          legend=dict(
                              x=0,
                              y=1,
                              traceorder="normal",
                              font=dict(
                                family="sans-serif",
                                size=12,
                                color="black"),
                              bgcolor="LightSteelBlue",
                              bordercolor="Black",
                              borderwidth=2)
                          )

        fig.show(renderer=VisualizeParams.renderer)

    def getBPM(self, winsize=5):
        """
        Get the BPM signal extracted from the ground truth BVP signal.
        """
        self.spectrogram(winsize)
        return self.bpm, self.times


class BPMcuda:
    """
    This class transforms a BVP signal in a BPM signal using GPU.

    BVP signal must be a float32 cupy.ndarray with shape [num_estimators, num_frames].
    """
    def __init__(self, data, fps, startTime=0, minHz=0.65, maxHz=4., verb=False):
        """
        Input 'data' is a BVP signal defined as a float32 cupy.ndarray with shape [num_estimators, num_frames]
        """
        self.nFFT = 2048//1  # freq. resolution for STFTs
        if len(data.shape) == 1:
            self.data = data.reshape(1, -1)  # 2D array raw-wise
        else:
            self.data = data
        self.fps = fps                       # sample rate
        self.startTime = startTime
        self.verb = verb
        self.minHz = minHz
        self.maxHz = maxHz
        self.gpuData = cupy.asarray(
            [self.fps, self.nFFT, self.minHz, self.maxHz])
        self.gmodel = Model(gaussian, independent_vars=['x', 'mu', 'a'])

    def BVP_to_BPM(self):
        """
        Return the BPM signal as a float32 cupy.ndarray with shape [num_estimators, ]. Remember that
        the value returned is on GPU, use cupy.asnumpy() to transform it to a Numpy ndarray.

        This method use the Welch's method to estimate the spectral density of the BVP signal,
        then it chooses as BPM the maximum Amplitude frequency.
        """
        # -- interpolation for less than 256 samples
        _, n = self.data.shape
        if self.data.shape[0] == 0:
            return cupy.float32(0.0)
        Pfreqs, Power = Welch_cuda(self.data, self.gpuData[0], self.gpuData[2], self.gpuData[3], self.gpuData[1])
        # -- BPM estimate 
        Pmax = cupy.argmax(Power, axis=1)  # power max
        return Pfreqs[Pmax.squeeze()]

    
    def BVP_to_BPM_PSD_clustering(self, out_fact=1):
        """
        Return the BPM signal as a numpy.float32. 

        TODO: cambiare descrizione.
        This method use the Welch's method to estimate the spectral density of the BVP signal; in case
        of multiple estimators the method sum all the Power Spectums, then it chooses as BPM the 
        maximum Amplitude frequency.
        """
        # -- interpolation for less than 256 samples
        if self.data.shape[0] == 0:
            return np.float32(0.0)
        Pfreqs, Power = Welch_cuda(self.data, self.fps, self.minHz, self.maxHz, self.nFFT)
        # F are frequencies, PSD is Power Spectrum Density of all estimators
        F = cupy.asnumpy(Pfreqs)
        PSD = cupy.asnumpy(Power)
        # Less then 3 estimators -  don't need clustering
        # we choose the maximum Amplitude frequency
        if PSD.shape[0] < 3:
          IDmax = np.unravel_index(np.argmax(PSD, axis=None), PSD.shape)
          Fmax = F[IDmax[1]]
          return np.float32(Fmax)

        W = pairwise_distances(PSD, PSD, metric='cosine')
        theta = circle_clustering(W, eps=0.01)
        # bi-partition, sum and normalization
        PSD0, PSD1, PSD_outlayers, median_PSD0, median_PSD1 = optimize_partition(theta, out_fact=out_fact)

        P0 = np.sum(PSD[PSD0,:], axis=0)
        max = np.max(P0, axis=0)
        max = np.expand_dims(max, axis=0)
        P0 = np.squeeze(np.divide(P0, max))

        P1 = np.sum(PSD[PSD1,:], axis=0)
        max = np.max(P1, axis=0)
        max = np.expand_dims(max, axis=0)
        P1 = np.squeeze(np.divide(P1, max))

        # peaks
        peak0_idx = np.argmax(P0) 
        P0_max = P0[peak0_idx]
        F0 = F[peak0_idx]

        peak1_idx = np.argmax(P1) 
        P1_max = P1[peak1_idx]
        F1 = F[peak1_idx]
    
        # Gaussian fitting
        result0, G1, sigma0 = gaussian_fit(self.gmodel, P0, F, F0, 1)  # Gaussian fit 
        result1, G2, sigma1 = gaussian_fit(self.gmodel, P1, F, F1, 1)  # Gaussian fit 
        chis0 = result0.chisqr
        chis1 = result1.chisqr
        rchis0 = result0.redchi
        rchis1 = result1.redchi
        aic0 = result0.aic
        aic1 = result1.aic
        bic0 = result0.bic
        bic1 = result1.bic

        # ranking
        rankP0 = 0
        rankP1 = 0
        rankP0 = rankP0 + 1 if chis0 < chis1 else rankP0
        rankP1 = rankP1 + 1 if chis0 >= chis1 else rankP1
        rankP0 = rankP0 + 1 if rchis0 < rchis1 else rankP0
        rankP1 = rankP1 + 1 if rchis0 >= rchis1 else rankP1
        rankP0 = rankP0 + 1 if aic0 < aic1 else rankP0
        rankP1 = rankP1 + 1 if aic0 >= aic1 else rankP1
        rankP0 = rankP0 + 1 if bic0 < bic1 else rankP0
        rankP1 = rankP1 + 1 if bic0 >= bic1 else rankP1
        rankP0 = rankP0 + 1 if sigma0 < sigma1 else rankP0
        rankP1 = rankP1 + 1 if sigma0 >= sigma1 else rankP1

        # best fit
        bpm = None
        if rankP0 > rankP1:
            bpm = F0
        else:
            bpm = F1

        return np.float32(bpm)

class BPM:
    """
    This class transforms a BVP signal in a BPM signal using CPU.

    BVP signal must be a float32 numpy.ndarray with shape [num_estimators, num_frames].
    """
    def __init__(self, data, fps, startTime=0, minHz=0.65, maxHz=4., verb=False):
        """
        Input 'data' is a BVP signal defined as a float32 Numpy.ndarray with shape [num_estimators, num_frames]
        """
        self.nFFT = 2048//1  # freq. resolution for STFTs
        if len(data.shape) == 1:
            self.data = data.reshape(1, -1)  # 2D array raw-wise
        else:
            self.data = data
        self.fps = fps                       # sample rate
        self.startTime = startTime
        self.verb = verb
        self.minHz = minHz
        self.maxHz = maxHz
        # PSD_clustering needs a gaussain fitting model
        self.gmodel = Model(gaussian, independent_vars=['x', 'mu', 'a'])

    def BVP_to_BPM(self):
        """
        Return the BPM signal as a float32 Numpy.ndarray with shape [num_estimators, ].

        This method use the Welch's method to estimate the spectral density of the BVP signal,
        then it chooses as BPM the maximum Amplitude frequency.
        """
        if self.data.shape[0] == 0:
            return np.float32(0.0)
        Pfreqs, Power = Welch(self.data, self.fps, self.minHz, self.maxHz, self.nFFT)
        # -- BPM estimate
        Pmax = np.argmax(Power, axis=1)  # power max
        return Pfreqs[Pmax.squeeze()]

    def BVP_to_BPM_PSD_clustering(self, out_fact=1):
        """
        Return the BPM signal as a numpy.float32. 

        TODO: cambiare descrizione.
        This method use the Welch's method to estimate the spectral density of the BVP signal; in case
        of multiple estimators the method sum all the Power Spectums, then it chooses as BPM the 
        maximum Amplitude frequency.
        """
        # -- interpolation for less than 256 samples
        if self.data.shape[0] == 0:
            return np.float32(0.0)
        Pfreqs, Power = Welch(self.data, self.fps, self.minHz, self.maxHz, self.nFFT)
        # F are frequencies, PSD is Power Spectrum Density of all estimators
        F = Pfreqs
        PSD = Power
        # Less then 3 estimators -  don't need clustering
        # we choose the maximum Amplitude frequency
        if PSD.shape[0] < 3:
          IDmax = np.unravel_index(np.argmax(PSD, axis=None), PSD.shape)
          Fmax = F[IDmax[1]]
          return np.float32(Fmax)

        W = pairwise_distances(PSD, PSD, metric='cosine')
        theta = circle_clustering(W, eps=0.01)
        # bi-partition, sum and normalization
        PSD0, PSD1, PSD_outlayers, median_PSD0, median_PSD1 = optimize_partition(theta, out_fact=out_fact)

        P0 = np.sum(PSD[PSD0,:], axis=0)
        max = np.max(P0, axis=0)
        max = np.expand_dims(max, axis=0)
        P0 = np.squeeze(np.divide(P0, max))

        P1 = np.sum(PSD[PSD1,:], axis=0)
        max = np.max(P1, axis=0)
        max = np.expand_dims(max, axis=0)
        P1 = np.squeeze(np.divide(P1, max))

        # peaks
        peak0_idx = np.argmax(P0) 
        P0_max = P0[peak0_idx]
        F0 = F[peak0_idx]

        peak1_idx = np.argmax(P1) 
        P1_max = P1[peak1_idx]
        F1 = F[peak1_idx]
    
        # Gaussian fitting
        result0, G1, sigma0 = gaussian_fit(self.gmodel, P0, F, F0, 1)  # Gaussian fit 
        result1, G2, sigma1 = gaussian_fit(self.gmodel, P1, F, F1, 1)  # Gaussian fit 
        chis0 = result0.chisqr
        chis1 = result1.chisqr
        rchis0 = result0.redchi
        rchis1 = result1.redchi
        aic0 = result0.aic
        aic1 = result1.aic
        bic0 = result0.bic
        bic1 = result1.bic

        # ranking
        rankP0 = 0
        rankP1 = 0
        rankP0 = rankP0 + 1 if chis0 < chis1 else rankP0
        rankP1 = rankP1 + 1 if chis0 >= chis1 else rankP1
        rankP0 = rankP0 + 1 if rchis0 < rchis1 else rankP0
        rankP1 = rankP1 + 1 if rchis0 >= rchis1 else rankP1
        rankP0 = rankP0 + 1 if aic0 < aic1 else rankP0
        rankP1 = rankP1 + 1 if aic0 >= aic1 else rankP1
        rankP0 = rankP0 + 1 if bic0 < bic1 else rankP0
        rankP1 = rankP1 + 1 if bic0 >= bic1 else rankP1
        rankP0 = rankP0 + 1 if sigma0 < sigma1 else rankP0
        rankP1 = rankP1 + 1 if sigma0 >= sigma1 else rankP1

        # best fit
        bpm = None
        if rankP0 > rankP1:
            bpm = F0
        else:
            bpm = F1 

        return np.float32(bpm)


# --------------------------------------------------------------------- #


def multi_est_BPM_median(bpms):
    """
    This method is used for computing the median of a multi-estimators BPM windowed signal.

    Args:
        bpms (list): list of lenght num_windows of BPM signals, each defined as a Numpy.ndarray with shape [num_estimators, ],
            or 1D Numpy.ndarray in case of a single estimator;

    Returns: 
        The median of the multi-estimators BPM signal defined as a float32 Numpy.ndarray with shape [num_windows,]; if a window 
        has num_estimators == 0, then the median value is set to 0.0 .
        The Median Absolute Deviation (MAD) of the multi-estimators BPM signal
    """
    median_bpms = np.zeros((len(bpms),))
    MAD = np.zeros((len(bpms),))
    i = 0
    for bpm in bpms:
        if len(bpm.shape) > 0 and bpm.shape[0] == 0:
            median_bpms[i] = np.float32(0.0)
            MAD[i] = 0.
        else:
            median_bpms[i] = np.float32(np.median(bpm))
            if len(bpm.shape) > 0:
                MAD[i] = np.float32(mad(bpm))                
        i += 1

    return median_bpms, MAD


def BVP_to_BPM(bvps, fps, minHz=0.65, maxHz=4.):
    """
    Transform a BVP windowed signal in a BPM signal using CPU.

    This method use the Welch's method to estimate the spectral density of the BVP signal,
    then it chooses as BPM the maximum Amplitude frequency.

    Args:
        bvps (list): list of length num_windows of BVP signal defined as float32 Numpy.ndarray with shape [num_estimators, num_frames].
        fps (float): frames per seconds.
        minHz (float): frequency in Hz used to isolate a specific subband [minHz, maxHz] (esclusive).
        maxHz (float): frequency in Hz used to isolate a specific subband [minHz, maxHz] (esclusive).

    Returns:
        A list of length num_windows of BPM signals defined as a float32 Numpy.ndarray with shape [num_estimators, ].
        If any BPM can't be found in a window, then the ndarray has num_estimators == 0.
        
    """
    bpms = []
    obj = None
    for bvp in bvps:
        if obj is None:
            obj = BPM(bvp, fps, minHz=minHz, maxHz=maxHz)
        else:
            obj.data = bvp
        bpm_es = obj.BVP_to_BPM()
        bpms.append(bpm_es)
    return bpms


def BVP_to_BPM_cuda(bvps, fps, minHz=0.65, maxHz=4.):
    """
    Transform a BVP windowed signal in a BPM signal using GPU.

    This method use the Welch's method to estimate the spectral density of the BVP signal,
    then it chooses as BPM the maximum Amplitude frequency.

    Args:
        bvps (list): list of length num_windows of BVP signal defined as float32 Numpy.ndarray with shape [num_estimators, num_frames].
        fps (float): frames per seconds.
        minHz (float): frequency in Hz used to isolate a specific subband [minHz, maxHz] (esclusive).
        maxHz (float): frequency in Hz used to isolate a specific subband [minHz, maxHz] (esclusive).

    Returns:
        A list of length num_windows of BPM signals defined as a float32 Numpy.ndarray with shape [num_estimators, ].
        If any BPM can't be found in a window, then the ndarray has num_estimators == 0.
        
    """
    bpms = []
    obj = None
    for bvp in bvps:
        bvp_device = cupy.asarray(bvp)
        if obj is None:
            obj = BPMcuda(bvp_device, fps, minHz=minHz, maxHz=maxHz)
        else:
            obj.data = bvp_device
        bpm_es = obj.BVP_to_BPM()
        bpm_es = cupy.asnumpy(bpm_es)
        bpms.append(bpm_es)
    return bpms


def BVP_to_BPM_PSD_clustering(bvps, fps, minHz=0.65, maxHz=4., out_fact=1):
    """
    Transform a BVP windowed signal in a BPM signal using CPU.

    TODO: riscrivere descrizione
    This method use the Welch's method to estimate the spectral density of the BVP signal; in case
    of multiple estimators the method sum all the Power Spectums, then it chooses as BPM the 
    maximum Amplitude frequency.

    Args:
        bvps (list): list of length num_windows of BVP signal defined as float32 Numpy.ndarray with shape [num_estimators, num_frames].
        fps (float): frames per seconds.
        minHz (float): frequency in Hz used to isolate a specific subband [minHz, maxHz] (esclusive).
        maxHz (float): frequency in Hz used to isolate a specific subband [minHz, maxHz] (esclusive).

    Returns:
        A list of length num_windows of BPM signals defined as a numpy.float32.
        If any BPM can't be found in a window, then the BPM is 0.0.
        
    """
    bpms = []
    obj = None
    for bvp in bvps:
        if obj is None:
            obj = BPM(bvp, fps, minHz=minHz, maxHz=maxHz)
        else:
            obj.data = bvp
        bpm_es = obj.BVP_to_BPM_PSD_clustering(out_fact=out_fact)
        bpms.append(bpm_es)
    return bpms


def BVP_to_BPM_PSD_clustering_cuda(bvps, fps, minHz=0.65, maxHz=4., out_fact=1):
    """
    Transform a BVP windowed signal in a BPM signal using GPU.

    TODO: riscrivere descrizione
    This method use the Welch's method to estimate the spectral density of the BVP signal; in case
    of multiple estimators the method sum all the Power Spectums, then it chooses as BPM the 
    maximum Amplitude frequency.

    Args:
        bvps (list): list of length num_windows of BVP signal defined as float32 Numpy.ndarray with shape [num_estimators, num_frames].
        fps (float): frames per seconds.
        minHz (float): frequency in Hz used to isolate a specific subband [minHz, maxHz] (esclusive).
        maxHz (float): frequency in Hz used to isolate a specific subband [minHz, maxHz] (esclusive).

    Returns:
        A list of length num_windows of BPM signals defined as a numpy.float32.
        If any BPM can't be found in a window, then the BPM is 0.0.
        
    """
    bpms = []
    obj = None
    for bvp in bvps:
        if obj is None:
            obj = BPMcuda(bvp, fps, minHz=minHz, maxHz=maxHz)
        else:
            obj.data = bvp
        bpm_es = obj.BVP_to_BPM_PSD_clustering(out_fact=out_fact)
        bpms.append(bpm_es)
    return bpms