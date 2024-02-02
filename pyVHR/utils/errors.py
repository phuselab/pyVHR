import numpy as np
import plotly.graph_objects as go
from pyVHR.plot.visualize import VisualizeParams
from pyVHR.BPM.utils import Welch
from pyVHR.extraction.utils import sliding_straded_win_idx

def getErrors(bvps, fps, bpmES, bpmGT, timesES, timesGT):
    """ Computes various error/quality measures"""
    if type(bpmES) == list:
        bpmES = np.expand_dims(bpmES, axis=0)
    if type(bpmES) == np.ndarray:
        if len(bpmES.shape) == 1:
            bpmES = np.expand_dims(bpmES, axis=0)

    RMSE = RMSEerror(bpmES, bpmGT, timesES, timesGT)
    MAE = MAEerror(bpmES, bpmGT, timesES, timesGT)
    MAX = MAXError(bpmES, bpmGT, timesES, timesGT)
    PCC = PearsonCorr(bpmES, bpmGT, timesES, timesGT)
    CCC = LinCorr(bpmES, bpmGT, timesES, timesGT)
    SNR = get_SNR(bvps, fps, bpmGT, timesES)

    return RMSE, MAE, MAX, PCC, CCC, SNR


def RMSEerror(bpmES, bpmGT, timesES=None, timesGT=None):
    """ Computes RMSE """

    diff = bpm_diff(bpmES, bpmGT, timesES, timesGT)
    n, m = diff.shape  # n = num channels, m = bpm length
    df = np.zeros(n)
    for j in range(m):
        for c in range(n):
            df[c] += np.power(diff[c, j], 2)

    # -- final RMSE
    RMSE = np.sqrt(df/m)
    return RMSE


def MAEerror(bpmES, bpmGT, timesES=None, timesGT=None):
    """ Computes MAE """

    diff = bpm_diff(bpmES, bpmGT, timesES, timesGT)
    n, m = diff.shape  # n = num channels, m = bpm length
    df = np.sum(np.abs(diff), axis=1)

    # -- final MAE
    MAE = df/m
    return MAE


def MAXError(bpmES, bpmGT, timesES=None, timesGT=None):
    """ computes MAX """

    diff = bpm_diff(bpmES, bpmGT, timesES, timesGT)
    n, m = diff.shape  # n = num channels, m = bpm length
    df = np.max(np.abs(diff), axis=1)

    # -- final MAX
    MAX = df
    return MAX


def PearsonCorr(bpmES, bpmGT, timesES=None, timesGT=None):
    """ Computes PCC """
    from scipy import stats

    diff = bpm_diff(bpmES, bpmGT, timesES, timesGT)
    n, m = diff.shape  # n = num channels, m = bpm length
    CC = np.zeros(n)
    for c in range(n):
        # -- corr
        r, p = stats.pearsonr(diff[c, :]+bpmES[c, :], bpmES[c, :])
        CC[c] = r
    return CC


def LinCorr(bpmES, bpmGT, timesES=None, timesGT=None):
    """ Computes CCC """
    diff = bpm_diff(bpmES, bpmGT, timesES, timesGT)
    n, m = diff.shape  # n = num channels, m = bpm length
    CCC = np.zeros(n)
    for c in range(n):
        # -- Lin's Concordance Correlation Coefficient
        ccc = concordance_correlation_coefficient(bpmES[c, :], diff[c, :]+bpmES[c, :])
        CCC[c] = ccc
    return CCC


def printErrors(RMSE, MAE, MAX, PCC, CCC, SNR):
    print("\n    * Errors: RMSE = %.2f, MAE = %.2f, MAX = %.2f, PCC = %.2f, CCC = %.2f, SNR = %.2f" %
          (RMSE, MAE, MAX, PCC, CCC, SNR))


def displayErrors(bpmES, bpmGT, timesES=None, timesGT=None):
    """"Plots errors"""
    if type(bpmES) == list:
        bpmES = np.expand_dims(bpmES, axis=0)
    if type(bpmES) == np.ndarray:
        if len(bpmES.shape) == 1:
            bpmES = np.expand_dims(bpmES, axis=0)


    if (timesES is None) or (timesGT is None):
        timesES = np.arange(m)
        timesGT = timesES

    diff = bpm_diff(bpmES, bpmGT, timesES, timesGT)
    n, m = diff.shape  # n = num channels, m = bpm length
    df = np.abs(diff)
    dfMean = np.around(np.mean(df, axis=1), 1)

    # -- plot errors
    fig = go.Figure()
    name = 'Ch 1 (µ = ' + str(dfMean[0]) + ' )'
    fig.add_trace(go.Scatter(
        x=timesES, y=df[0, :], name=name, mode='lines+markers'))
    if n > 1:
        name = 'Ch 2 (µ = ' + str(dfMean[1]) + ' )'
        fig.add_trace(go.Scatter(
            x=timesES, y=df[1, :], name=name, mode='lines+markers'))
        name = 'Ch 3 (µ = ' + str(dfMean[2]) + ' )'
        fig.add_trace(go.Scatter(
            x=timesES, y=df[2, :], name=name, mode='lines+markers'))
    fig.update_layout(xaxis_title='Times (sec)',
                      yaxis_title='MAE', showlegend=True)
    fig.show(renderer=VisualizeParams.renderer)

    # -- plot bpm Gt and ES
    fig = go.Figure()
    GTmean = np.around(np.mean(bpmGT), 1)
    name = 'GT (µ = ' + str(GTmean) + ' )'
    fig.add_trace(go.Scatter(x=timesGT, y=bpmGT,
                             name=name, mode='lines+markers'))
    ESmean = np.around(np.mean(bpmES[0, :]), 1)
    name = 'ES1 (µ = ' + str(ESmean) + ' )'
    fig.add_trace(go.Scatter(
        x=timesES, y=bpmES[0, :], name=name, mode='lines+markers'))
    if n > 1:
        ESmean = np.around(np.mean(bpmES[1, :]), 1)
        name = 'ES2 (µ = ' + str(ESmean) + ' )'
        fig.add_trace(go.Scatter(
            x=timesES, y=bpmES[1, :], name=name, mode='lines+markers'))
        ESmean = np.around(np.mean(bpmES[2, :]), 1)
        name = 'E3 (µ = ' + str(ESmean) + ' )'
        fig.add_trace(go.Scatter(
            x=timesES, y=bpmES[2, :], name=name, mode='lines+markers'))

    fig.update_layout(xaxis_title='Times (sec)',
                      yaxis_title='BPM', showlegend=True)
    fig.show(renderer=VisualizeParams.renderer)


def bpm_diff(bpmES, bpmGT, timesES=None, timesGT=None):
    n, m = bpmES.shape  # n = num channels, m = bpm length

    if (timesES is None) or (timesGT is None):
        timesES = np.arange(m)
        timesGT = timesES

    diff = np.zeros((n, m))
    for j in range(m):
        t = timesES[j]
        i = np.argmin(np.abs(t-timesGT))
        for c in range(n):
            diff[c, j] = bpmGT[i]-bpmES[c, j]
    return diff


def concordance_correlation_coefficient(bpm_true, bpm_pred):
    cor=np.corrcoef(bpm_true, bpm_pred)[0][1]
    mean_true = np.mean(bpm_true)
    mean_pred = np.mean(bpm_pred)
    
    var_true = np.var(bpm_true)
    var_pred = np.var(bpm_pred)
    
    sd_true = np.std(bpm_true)
    sd_pred = np.std(bpm_pred)
    
    numerator = 2*cor*sd_true*sd_pred
    denominator = var_true + var_pred + (mean_true - mean_pred)**2

    return numerator/denominator


def get_SNR(bvps, fps, reference_hrs, timesES):
    '''Computes the signal-to-noise ratio of the BVP
    signals according to the method by -- de Haan G. et al., IEEE Transactions on Biomedical Engineering (2013).
    SNR calculated as the ratio (in dB) of power contained within +/- 0.1 Hz
    of the reference heart rate frequency and +/- 0.2 of its first
    harmonic and sum of all other power between 0.5 and 4 Hz.
    Adapted from https://github.com/danmcduff/iphys-toolbox/blob/master/tools/bvpsnr.m
    '''
   
    interv1 = 0.2*60
    interv2 = 0.2*60
    NyquistF = fps/2.;
    FResBPM = 0.5
    nfft = np.ceil((60*2*NyquistF)/FResBPM)
    SNRs = []
    for idx, bvp in enumerate(bvps):
        curr_ref = reference_hrs[idx]
        pfreqs, power = Welch(bvp, fps, nfft=nfft)
        GTMask1 = np.logical_and(pfreqs>=curr_ref-interv1, pfreqs<=curr_ref+interv1)
        GTMask2 = np.logical_and(pfreqs>=(curr_ref*2)-interv2, pfreqs<=(curr_ref*2)+interv2)
        GTMask = np.logical_or(GTMask1, GTMask2)
        FMask = np.logical_not(GTMask)
        win_snr = []
        for i in range(len(power)):
            p = power[i,:]
            SPower = np.sum(p[GTMask])
            allPower = np.sum(p[FMask])
            snr = 10*np.log10(SPower/allPower)
            win_snr.append(snr)
        SNRs.append(np.median(win_snr))
    return np.array([np.mean(SNRs)])

def BVP_windowing(bvp, wsize, fps, stride=1):
  """ Performs BVP signal windowing

    Args:
      bvp (list/array): full BVP signal
      wsize     (float): size of the window (in seconds)
      fps       (float): frames per seconds
      stride    (float): stride (in seconds)

    Returns:
      bvp_win (list): windowed BVP signal
      timesES (list): times of (centers) windows 
  """
  
  bvp = np.array(bvp).squeeze()
  block_idx, timesES = sliding_straded_win_idx(bvp.shape[0], wsize, stride, fps)
  bvp_win  = []
  for e in block_idx:
      st_frame = int(e[0])
      end_frame = int(e[-1])
      wind_signal = np.copy(bvp[st_frame: end_frame+1])
      bvp_win.append(wind_signal[np.newaxis, :])

  return bvp_win, timesES

def _plot_PSD_snr(pfreqs, p, curr_ref, interv1, interv2):
    import matplotlib.pyplot as plt
    import numpy as np
    plt.plot(pfreqs, np.squeeze(p))
    x1 = pfreqs[np.argmin(np.abs(pfreqs-curr_ref))]
    x2 = pfreqs[np.argmin(np.abs(pfreqs-curr_ref))]
    y1 = 0
    y2 = p[np.argmin(np.abs(pfreqs-curr_ref))]
    plt.plot([x1, x2], [y1, y2], color='r', linestyle='-', linewidth=2)
    x1 = pfreqs[np.argmin(np.abs(pfreqs-curr_ref*2))]
    x2 = pfreqs[np.argmin(np.abs(pfreqs-curr_ref*2))]
    y1 = 0
    y2 = p[np.argmin(np.abs(pfreqs-curr_ref*2))]
    plt.plot([x1, x2], [y1, y2], color='r', linestyle='-', linewidth=2)
    x1 = pfreqs[np.argmin(np.abs(pfreqs-curr_ref-interv1))]
    x2 = pfreqs[np.argmin(np.abs(pfreqs-curr_ref-interv1))]
    y1 = 0
    y2 = p[np.argmin(np.abs(pfreqs-curr_ref-interv1))]
    plt.plot([x1, x2], [y1, y2], color='k', linestyle='-', linewidth=2)
    x1 = pfreqs[np.argmin(np.abs(pfreqs-curr_ref+interv1))]
    x2 = pfreqs[np.argmin(np.abs(pfreqs-curr_ref+interv1))]
    y1 = 0
    y2 = p[np.argmin(np.abs(pfreqs-curr_ref+interv1))]
    plt.plot([x1, x2], [y1, y2], color='k', linestyle='-', linewidth=2)
    x1 = pfreqs[np.argmin(np.abs(pfreqs-curr_ref*2-interv1))]
    x2 = pfreqs[np.argmin(np.abs(pfreqs-curr_ref*2-interv1))]
    y1 = 0
    y2 = p[np.argmin(np.abs(pfreqs-curr_ref*2-interv1))]
    plt.plot([x1, x2], [y1, y2], color='k', linestyle='-', linewidth=2)
    x1 = pfreqs[np.argmin(np.abs(pfreqs-curr_ref*2+interv1))]
    x2 = pfreqs[np.argmin(np.abs(pfreqs-curr_ref*2+interv1))]
    y1 = 0
    y2 = p[np.argmin(np.abs(pfreqs-curr_ref*2+interv1))]
    plt.plot([x1, x2], [y1, y2], color='k', linestyle='-', linewidth=2)
    plt.show()
