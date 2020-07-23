import numpy as np
import plotly.graph_objects as go
from pyVHR.signals.bvp import BVPsignal

def getErrors(bpmES, bpmGT, timesES, timesGT):
    RMSE = RMSEerror(bpmES, bpmGT, timesES, timesGT)
    MAE = MAEerror(bpmES, bpmGT, timesES, timesGT)
    MAX = MAXError(bpmES, bpmGT, timesES, timesGT)
    PCC = PearsonCorr(bpmES, bpmGT, timesES, timesGT)
    return RMSE, MAE, MAX, PCC

def RMSEerror(bpmES, bpmGT, timesES=None, timesGT=None):
    """ RMSE: """

    diff = bpm_diff(bpmES, bpmGT, timesES, timesGT)
    n,m = diff.shape  # n = num channels, m = bpm length
    df = np.zeros(n)
    for j in range(m):
        for c in range(n):
            df[c] += np.power(diff[c,j],2)

    # -- final RMSE
    RMSE = np.sqrt(df/m)
    return RMSE

def MAEerror(bpmES, bpmGT, timesES=None, timesGT=None):
    """ MAE: """

    diff = bpm_diff(bpmES, bpmGT, timesES, timesGT)
    n,m = diff.shape  # n = num channels, m = bpm length
    df = np.sum(np.abs(diff),axis=1)

    # -- final MAE
    MAE = df/m
    return MAE

def MAXError(bpmES, bpmGT, timesES=None, timesGT=None):
    """ MAE: """

    diff = bpm_diff(bpmES, bpmGT, timesES, timesGT)
    n,m = diff.shape  # n = num channels, m = bpm length
    df = np.max(np.abs(diff),axis=1)

    # -- final MAE
    MAX = df
    return MAX

def PearsonCorr(bpmES, bpmGT, timesES=None, timesGT=None):
    from scipy import stats

    diff = bpm_diff(bpmES, bpmGT, timesES, timesGT)
    n,m = diff.shape  # n = num channels, m = bpm length
    CC = np.zeros(n)
    for c in range(n):
        # -- corr
        r,p = stats.pearsonr(diff[c,:]+bpmES[c,:],bpmES[c,:])
        CC[c] = r
    return CC

def printErrors(RMSE, MAE, MAX, PCC):
    print("\n    * Errors: RMSE = %.2f, MAE = %.2f, MAX = %.2f, PCC = %.2f" %(RMSE,MAE,MAX,PCC))

def displayErrors(bpmES, bpmGT, timesES=None, timesGT=None):
    
    if (timesES is None) or (timesGT is None):
        timesES = np.arange(m)
        timesGT = timesES
        
    diff = bpm_diff(bpmES, bpmGT, timesES, timesGT)
    n,m = diff.shape  # n = num channels, m = bpm length
    df = np.abs(diff)
    dfMean = np.around(np.mean(df,axis=1),1)

    # -- plot errors
    fig = go.Figure()
    name = 'Ch 1 (µ = ' + str(dfMean[0])+ ' )'
    fig.add_trace(go.Scatter(x=timesES, y=df[0,:], name=name, mode='lines+markers'))
    if n > 1:
        name = 'Ch 2 (µ = ' + str(dfMean[1])+ ' )'
        fig.add_trace(go.Scatter(x=timesES, y=df[1,:], name=name, mode='lines+markers'))
        name = 'Ch 3 (µ = ' + str(dfMean[2])+ ' )'
        fig.add_trace(go.Scatter(x=timesES, y=df[2,:], name=name, mode='lines+markers'))
    fig.update_layout(xaxis_title='Times (sec)', yaxis_title='MAE', showlegend=True)
    fig.show()

    # -- plot bpm Gt and ES 
    fig = go.Figure()
    GTmean = np.around(np.mean(bpmGT),1)
    name = 'GT (µ = ' + str(GTmean)+ ' )'
    fig.add_trace(go.Scatter(x=timesGT, y=bpmGT, name=name, mode='lines+markers'))
    ESmean = np.around(np.mean(bpmES[0,:]),1)
    name = 'ES1 (µ = ' + str(ESmean)+ ' )'
    fig.add_trace(go.Scatter(x=timesES, y=bpmES[0,:], name=name, mode='lines+markers'))
    if n > 1:
        ESmean = np.around(np.mean(bpmES[1,:]),1)
        name = 'ES2 (µ = ' + str(ESmean)+ ' )'
        fig.add_trace(go.Scatter(x=timesES, y=bpmES[1,:], name=name, mode='lines+markers'))
        ESmean = np.around(np.mean(bpmES[2,:]),1)
        name = 'E3 (µ = ' + str(ESmean)+ ' )'
        fig.add_trace(go.Scatter(x=timesES, y=bpmES[2,:], name=name, mode='lines+markers'))

    fig.update_layout(xaxis_title='Times (sec)', yaxis_title='BPM', showlegend=True)
    fig.show()


def bpm_diff(bpmES, bpmGT, timesES=None, timesGT=None):
    n,m = bpmES.shape  # n = num channels, m = bpm length

    if (timesES is None) or (timesGT is None):
        timesES = np.arange(m)
        timesGT = timesES
            
    diff = np.zeros((n,m))
    for j in range(m):
        t = timesES[j]
        i = np.argmin(np.abs(t-timesGT))
        for c in range(n):
            diff[c,j] = bpmGT[i]-bpmES[c,j]
    return diff
