from sklearn.metrics import silhouette_score, pairwise_distances
from numpy import linalg as LA
from sklearn.cluster import KMeans
#from spherecluster import VonMisesFisherMixture
from sklearn.neighbors import KernelDensity
from lmfit import Model
from scipy.stats import iqr
import numpy as np
from scipy.signal import welch
import cusignal
import cupy

"""
This module contains usefull methods used in pyVHR.BPM.BPM and
pyVHR.plot.visualize.
"""

def Welch(bvps, fps, minHz=0.65, maxHz=4.0, nfft=2048):
    """
    This function computes Welch'method for spectral density estimation.

    Args:
        bvps(flaot32 numpy.ndarray): BVP signal as float32 Numpy.ndarray with shape [num_estimators, num_frames].
        fps (float): frames per seconds.
        minHz (float): frequency in Hz used to isolate a specific subband [minHz, maxHz] (esclusive).
        maxHz (float): frequency in Hz used to isolate a specific subband [minHz, maxHz] (esclusive).
        nfft (int): number of DFT points, specified as a positive integer.
    Returns:
        Sample frequencies as float32 numpy.ndarray, and Power spectral density or power spectrum as float32 numpy.ndarray.
    """
    _, n = bvps.shape
    if n < 256:
        seglength = n
        overlap = int(0.8*n)  # fixed overlapping
    else:
        seglength = 256
        overlap = 200
    # -- periodogram by Welch
    F, P = welch(bvps, nperseg=seglength, noverlap=overlap, fs=fps, nfft=nfft)
    F = F.astype(np.float32)
    P = P.astype(np.float32)
    # -- freq subband (0.65 Hz - 4.0 Hz)
    band = np.argwhere((F > minHz) & (F < maxHz)).flatten()
    Pfreqs = 60*F[band]
    Power = P[:, band]
    return Pfreqs, Power

def Welch_cuda(bvps, fps, minHz=0.65, maxHz=4.0, nfft=2048):
    """
    This function computes Welch'method for spectral density estimation on CUDA GPU.

    Args:
        bvps(flaot32 cupy.ndarray): BVP signal as float32 Numpy.ndarray with shape [num_estimators, num_frames].
        fps (cupy.float32): frames per seconds.
        minHz (cupy.float32): frequency in Hz used to isolate a specific subband [minHz, maxHz] (esclusive).
        maxHz (cupy.float32): frequency in Hz used to isolate a specific subband [minHz, maxHz] (esclusive).
        nfft (cupy.int32): number of DFT points, specified as a positive integer.
    Returns:
        Sample frequencies as float32 cupy.ndarray, and Power spectral density or power spectrum as float32 cupy.ndarray.
    """
    _, n = bvps.shape
    if n < 256:
        seglength = n
        overlap = int(0.8*n)  # fixed overlapping
    else:
        seglength = 256
        overlap = 200
    # -- periodogram by Welch
    F, P = cusignal.welch(bvps, nperseg=seglength,
                            noverlap=overlap, fs=fps, nfft=nfft)
    # -- freq subband (0.65 Hz - 4.0 Hz)
    band = cupy.argwhere((F > minHz) & (F < maxHz)).flatten()
    Pfreqs = 60*F[band]
    Power = P[:, band]
    return Pfreqs, Power

def circle_clustering(W, eps=0.01, theta0=None):
    """ Provides a partition of elements with distance matrix W
        Args:
            W (float): distance matrix w_ij = w_ji = dist(X_i, X_j)
        
        Returns:
            angles (float): the angles theta_i of elements in a circle representing X_i 
    """    
    # general vars
    PI = np.pi
    PI2 = 2*PI
    n = W.shape[0]

    # param check
    if theta0 is None:
        theta = PI2*np.random.rand(n)  # init. values in [0, 2*PI]
    else:
        theta = theta0

    # preliminar computations 
    sin_t = np.sin(theta)
    cos_t = np.cos(theta)
    C = np.dot(W, cos_t)
    S = np.dot(W, sin_t)

    # main loop
    ok = True
    rounds = 0
    while ok:
        ok = False
        rounds += 1
        nchanges = 0

        # loop on thetas
        for k in range(n):
            old = theta[k]

            # compute new theta_k, i.e. theta_k'
            theta[k] = np.arctan(S[k]/C[k])    # within [-PI/2, PI/2]
            if C[k] >= 0:
                theta[k] += PI
            elif S[k] >= 0:
                theta[k] += PI2

            # check condition
            if abs(old-theta[k]) > eps:
                ok = True
                nchanges += 1
                # update Ck & Sk by elementwise product and diff
                C += np.multiply(W[k,:], np.repeat(np.cos(theta[k]) - np.cos(old), n))
                S += np.multiply(W[k,:], np.repeat(np.sin(theta[k]) - np.sin(old), n)) 

    return theta

def optimize_partition(theta, opt_factor=.5):
  n = theta.shape[0]
  T = theta[:,None] - theta  # x[:,None] adds a second axis to the array
  T = np.cos(T) - np.eye(n)  # matrix of cosine diff
  
  # compute partitions P,Q
  P = []
  Q = []
  Tmean = np.mean(theta)
  for idx,th in enumerate(theta):
    if np.sign(np.sin(Tmean-th)) > 0:
      P.append(idx)
    else:
      Q.append(idx)

  # adjust partitions P,Q
  OK = False
  while OK:
    OK = False
    for i in range(n):
      if i in P:
        if np.sum(T[i,P]) < np.sum(T[i,Q]) and len(P) > 1:
          Q.append(i)
          P.remove(i)
          OK = True
      else: 
        if np.sum(T[i,P]) > np.sum(T[i,Q]) and len(Q) > 1:
          Q.remove(i)
          P.append(i)
          OK = True
    
  # pull out outliers from P and Q
  A = []
  B = []
  Z = []
  for i in P:
    A.append(np.sum(T[i,P]) / (T[i,P].shape[0]-1))
  
  M = np.max(A)
  M_idx = np.argmax(A)
  S = np.std(A)
  max_theta_P = theta[P[M_idx]] 
  L = M - opt_factor*S 

  # prune outliers in P
  for idx,i in enumerate(P):
    if M_idx != idx and A[idx] < L:
      Z.append(i)  

  for i in Q:
    B.append(np.sum(T[i,Q]) / (T[i,Q].shape[0]-1))
  
  M = np.max(B)
  M_idx = np.argmax(B)
  S = np.std(B)
  max_theta_Q = theta[Q[M_idx]] 
  L = M - opt_factor*S  

  # prune outliers in Q
  for idx,i in enumerate(Q):
    if M_idx != idx and B[idx] < L:
      Z.append(i)

  P = list(set(P) - set(Z))
  if len(P) == 0:
    print('ERROR empty list!')
  Q = list(set(Q) - set(Z))
  if len(Q) == 0:
    print('ERROR empty list!')

  return P, Q, Z, max_theta_P, max_theta_Q 
 
def gaussian(x,a,mu,sigma):
    return a*np.exp(-(x-mu)**2/(2*sigma**2))

def gaussian_fit(p, x, mu, max):
  gmodel = Model(gaussian, independent_vars=['x', 'mu', 'a'])
  result = gmodel.fit(p, x=x, a=max, mu=mu, sigma=1)
  sigma = result.params['sigma'].value
  g = gaussian(x, max, mu, sigma)
  return result, g, sigma

def PSD_SNR(PSD, f_peak, sigma, freqs):
  """ PSD estimate based SNR """
  
  ERR_mask = np.logical_or(freqs < f_peak-3*sigma, freqs > f_peak+3*sigma)
  SIG_mask = np.logical_not(ERR_mask)
  SIG_power = np.sum(PSD[SIG_mask]) # signal power
  ERR_power = np.sum(PSD[ERR_mask]) # noise power
  if ERR_power < 10e-8:
    SNR = 0    # it denotes anomaly
  else:
    SNR = SIG_power/ERR_power  # linear SNR
  return SNR, SIG_mask

def shrink(x, alpha=4):
    return np.multiply(x,1-np.exp(-alpha*x))