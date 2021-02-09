import numpy as np
from scipy.signal import find_peaks, stft, lfilter, butter, welch
from plotly.subplots import make_subplots
from plotly.colors import n_colors
import plotly.graph_objects as go
from scipy.interpolate import interp1d


class BVPsignal:
    """
        Manage (multi-channel, row-wise) BVP signals
    """
    nFFT = 2048  # freq. resolution for STFTs
    step = 1       # step in seconds

    def __init__(self, data, fs, startTime=0, minHz=0.75, maxHz=4., verb=False):
        if len(data.shape) == 1:
            self.data = data.reshape(1,-1) # 2D array raw-wise
        else:
            self.data = data
        self.numChls = self.data.shape[0]  # num  channels
        self.fs = fs                       # sample rate
        self.startTime = startTime
        self.verb = verb
        self.minHz = minHz
        self.maxHz = maxHz

    def getChunk(startTime, winsize=None, numSample=None):
        
        assert startTime >= self.startTime, "Start time error!"
        
        N = self.data.shape[1] 
        fs = self.fs
        Nstart = int(fs*startTime)
        
        # -- winsize > 0
        if winsize:
            stopTime = startTime + winsize
            Nstop = np.min([int(fs*stopTime),N])
            
        # -- numSample > 0
        if numSample:
            Nstop = np.min([numSample,N])
        
        return self.data[0,Nstart:Nstop]
        
    def hps(self, spect, d=3):
        
        if spect.ndim == 2:
            n_win = spect.shape[1]
            new_spect = np.zeros_like(spect)
            for w in range(n_win):
                curr_w = spect[:,w]
                w_down_z = np.zeros_like(curr_w)
                w_down = curr_w[::d]
                w_down_z[0:len(w_down)] = w_down
                w_hps = np.multiply(curr_w, w_down_z)
                new_spect[:, w] = w_hps
            return new_spect

        elif spect.ndim == 1:
            s_down_z = np.zeros_like(spect)
            s_down = spect[::d]
            s_down_z[0:len(s_down)] = s_down
            w_hps = np.multiply(spect, s_down_z)
            return w_hps

        else:
            raise ValueError("Wrong Dimensionality of the Spectrogram for the HPS")

    def spectrogram(self, winsize=5, use_hps=False):
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
        minHz = 0.75
        maxHz = 4.0
        band = np.argwhere((F > minHz) & (F < maxHz)).flatten()
        self.spect = np.abs(Z[band,:])     # spectrum magnitude
        self.freqs = 60*F[band]            # spectrum freq in bpm
        self.times = T                     # spectrum times

        if use_hps:
            spect_hps = self.hps(self.spect)
            # -- BPM estimate by spectrum
            self.bpm = self.freqs[np.argmax(spect_hps,axis=0)]
        else:
            # -- BPM estimate by spectrum
            self.bpm = self.freqs[np.argmax(self.spect,axis=0)]
        
    def getBPM(self, winsize=5):
        self.spectrogram(winsize, use_hps=False)
        return self.bpm, self.times
    
    def PSD2BPM(self, chooseBest=True, use_hps=False):
        """
            Compute power spectral density using Welch’s method and estimate
            BPMs from video frames
        """

        # -- interpolation for less than 256 samples
        c,n = self.data.shape
        if n < 256:
            seglength = n
            overlap = int(0.8*n)  # fixed overlapping
        else:
            seglength = 256
            overlap = 200
       
        # -- periodogram by Welch
        F, P = welch(self.data, nperseg=seglength, noverlap=overlap, window='hamming',fs=self.fs, nfft=self.nFFT)

        # -- freq subband (0.75 Hz - 4.0 Hz)
        band = np.argwhere((F > self.minHz) & (F < self.maxHz)).flatten()
        self.Pfreqs = 60*F[band]
        self.Power = P[:,band]
        
        # -- if c = 3 choose that with the best SNR
        if chooseBest:
            winner = 0
            lobes = self.PDSrippleAnalysis(ch=0)
            SNR = lobes[-1]/lobes[-2]
            if c == 3:
                lobes = self.PDSrippleAnalysis(ch=1)
                SNR1 = lobes[-1]/lobes[-2]
                if SNR1 > SNR:
                    SNR = SNR1
                    winner = 1
                lobes = self.PDSrippleAnalysis(ch=2)
                SNR1 = lobes[-1]/lobes[-2]
                if SNR1 > SNR:
                    SNR = SNR1
                    winner = 2    
            self.Power = self.Power[winner].reshape(1,-1)
        
        # TODO: eliminare?
        if use_hps:
            p = self.Power[0]
            phps = self.hps(p)
            '''import matplotlib.pyplot as plt
            plt.plot(p)
            plt.figure()
            plt.plot(phps)
            plt.show()'''
            Pmax = np.argmax(phps)  # power max
            self.bpm = np.array([self.Pfreqs[Pmax]])       # freq max

        else:
            # -- BPM estimate by PSD
            Pmax = np.argmax(self.Power, axis=1)  # power max
            self.bpm = self.Pfreqs[Pmax]       # freq max

        if '3' in str(self.verb):
            lobes = self.PDSrippleAnalysis()
            self.displayPSD(lobe1=lobes[-1], lobe2=lobes[-2])

    def autocorr(self):
        from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

        # TODO: to handle all channels
        x = self.data[0,:]
        plot_acf(x)
        plt.show()

        plot_pacf(x)
        plt.show()

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
        fig.add_trace(go.Scatter(x=t, y=bpm, name='Frequency Domain', line=dict(color='red', width=2)))

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
                       
        fig.show()

    def findPeaks(self, distance=None, height=None):
        
        # -- take the first channel
        x = self.data[0].flatten()
            
        if distance is None:
            distance = self.fs/2
        if height is None:
            height = np.mean(x)

        # -- find peaks with the specified params
        self.peaks, _ = find_peaks(x, distance=distance, height=height)
        
        self.peaksTimes = self.peaks/self.fs
        self.bpmPEAKS = 60.0/np.diff(self.peaksTimes)
        
    def plotBPMPeaks(self, height=None, width=None):
        """
            Plot the the BVP signal and peak marks
        """

        # -- find peaks  
        try:
            peaks = self.peaks
        except AttributeError:
            self.findPeaks()
            peaks = self.peaks
        
        #-- signals 
        y = self.data[0]
        n = y.shape[0]
        startTime  = self.startTime 
        stopTime = startTime+n/self.fs
        x = np.linspace(startTime, stopTime, num=n, endpoint=False)
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=x, y=y, name="BVP"))
        fig.add_trace(go.Scatter(x=x[peaks], y=y[peaks], mode='markers', name="Peaks"))

        if not height:
            height=400
        if not width:
            width=800

        fig.update_layout(height=height, width=width, title="BVP signal + peaks",
            font=dict(
                family="Courier New, monospace",
                size=14,
                color="#7f7f7f"))
        
        fig.show()
        
    def plot(self, title="BVP signal", height=400, width=800):
        """
            Plot the the BVP signal (multiple channels)
        """
      
        #-- signals 
        y = self.data
        c,n = y.shape
        startTime  = self.startTime 
        stopTime = startTime+n/self.fs
        x = np.linspace(startTime, stopTime, num=n, endpoint=False)
        
        fig = go.Figure()
        
        for i in range(c):
            name = "BVP " + str(i)
            fig.add_trace(go.Scatter(x=x, y=y[i], name=name))

        fig.update_layout(height=height, width=width, title=title,
            font=dict(
                family="Courier New, monospace",
                size=14,
                color="#7f7f7f"))
        fig.show()
        
    def displayPSD(self, ch=0, lobe1=None, lobe2=None, GT=None):
        """Show the periodogram(s) of the BVP signal for channel ch"""

        f = self.Pfreqs 
        P = self.Power[ch] 
                
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(x=f, y=P, name='PSD'))
        fig.update_layout(autosize=False, width=500, height=400)
        
        if lobe1 is not None and lobe2 is not None:
            L1 = lobe1
            L2 = lobe2
            # Add horiz. lobe peack lines
            fig.add_shape(type="line",x0=f[0], y0=L1, x1=f[-1], y1=L1,
                line=dict(color="LightSeaGreen", width=2, dash="dashdot"))
            fig.add_shape(type="line",x0=f[0], y0=L2, x1=f[-1], y1=L2,
                line=dict(color="SeaGreen", width=2, dash="dashdot"))
            tit = 'SNR = ' + str(np.round(L1/L2,2))
            fig.update_layout(title=tit)
            
        if GT is not None:
            # Add vertical GT line
            fig.add_shape(type="line",x0=GT, y0=0, x1=GT, y1=np.max(P),
                line=dict(color="DarkGray", width=2, dash="dash"))
            
        fig.show()
  
    def PDSrippleAnalysis(self, ch=0):
        # -- ripple analysis
        
        P = self.Power[ch].flatten()
        dP = np.gradient(P)
        n = len(dP)
        I = []; 
        i = 0
        while i < n:
            m = 0
            # -- positive gradient
            while (i < n) and (dP[i] > 0):
                m = max([m,P[i]])
                i += 1
            I.append(m)
            # -- skip negative gradient
            while (i < n) and (dP[i] < 0) : 
                i += 1
        lobes = np.sort(I)
        if len(lobes) < 2:
            lobes = np.array([lobes,0])
            
        return lobes