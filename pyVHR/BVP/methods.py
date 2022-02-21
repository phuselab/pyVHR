import cupy
import math
import time
import numpy as np
import torch
import os
from sklearn.decomposition import PCA
from pyVHR.BVP.utils import jadeR


"""
This module contains a collection of known rPPG methods.

rPPG METHOD SIGNATURE
An rPPG method must accept theese parameters:
    > signal -> RGB signal as float32 ndarray with shape [num_estimators, rgb_channels, num_frames], or a custom signal.
    > **kargs [OPTIONAL] -> usefull parameters passed to the filter method.
It must return a BVP signal as float32 ndarray with shape [num_estimators, num_frames].
"""


# ------------------------------------------------------------------------------------- #
#                                     rPPG METHODS                                      #
# ------------------------------------------------------------------------------------- #


def cpu_CHROM(signal):
    """
    CHROM method on CPU using Numpy.

    De Haan, G., & Jeanne, V. (2013). Robust pulse rate from chrominance-based rPPG. IEEE Transactions on Biomedical Engineering, 60(10), 2878-2886.
    """
    X = signal
    Xcomp = 3*X[:, 0] - 2*X[:, 1]
    Ycomp = (1.5*X[:, 0])+X[:, 1]-(1.5*X[:, 2])
    sX = np.std(Xcomp, axis=1)
    sY = np.std(Ycomp, axis=1)
    alpha = (sX/sY).reshape(-1, 1)
    alpha = np.repeat(alpha, Xcomp.shape[1], 1)
    bvp = Xcomp - np.multiply(alpha, Ycomp)
    return bvp


def cupy_CHROM(signal):
    """
    CHROM method on GPU using Cupy.

    De Haan, G., & Jeanne, V. (2013). Robust pulse rate from chrominance-based rPPG. IEEE Transactions on Biomedical Engineering, 60(10), 2878-2886.
    """
    X = signal
    Xcomp = 3*X[:, 0] - 2*X[:, 1]
    Ycomp = (1.5*X[:, 0])+X[:, 1]-(1.5*X[:, 2])
    sX = cupy.std(Xcomp, axis=1)
    sY = cupy.std(Ycomp, axis=1)
    alpha = (sX/sY).reshape(-1, 1)
    alpha = cupy.repeat(alpha, Xcomp.shape[1], 1)
    bvp = Xcomp - cupy.multiply(alpha, Ycomp)
    return bvp


def torch_CHROM(signal):
    """
    CHROM method on CPU using Torch.

    De Haan, G., & Jeanne, V. (2013). Robust pulse rate from chrominance-based rPPG. IEEE Transactions on Biomedical Engineering, 60(10), 2878-2886.
    """
    X = signal
    Xcomp = 3*X[:, 0] - 2*X[:, 1]
    Ycomp = (1.5*X[:, 0])+X[:, 1]-(1.5*X[:, 2])
    sX = torch.std(Xcomp, axis=1)
    sY = torch.std(Ycomp, axis=1)
    alpha = (sX/sY).reshape(-1, 1)
    alpha = torch.repeat_interleave(alpha, Xcomp.shape[1], 1)
    bvp = Xcomp - torch.mul(alpha, Ycomp)
    return bvp


def cpu_LGI(signal):
    """
    LGI method on CPU using Numpy.

    Pilz, C. S., Zaunseder, S., Krajewski, J., & Blazek, V. (2018). Local group invariance for heart rate estimation from face videos in the wild. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition Workshops (pp. 1254-1262).
    """
    X = signal
    U, _, _ = np.linalg.svd(X)
    S = U[:, :, 0]
    S = np.expand_dims(S, 2)
    sst = np.matmul(S, np.swapaxes(S, 1, 2))
    p = np.tile(np.identity(3), (S.shape[0], 1, 1))
    P = p - sst
    Y = np.matmul(P, X)
    bvp = Y[:, 1, :]
    return bvp


def cpu_POS(signal, **kargs):
    """
    POS method on CPU using Numpy.

    The dictionary parameters are: {'fps':float}.

    Wang, W., den Brinker, A. C., Stuijk, S., & de Haan, G. (2016). Algorithmic principles of remote PPG. IEEE Transactions on Biomedical Engineering, 64(7), 1479-1491. 
    """
    # Run the pos algorithm on the RGB color signal c with sliding window length wlen
    # Recommended value for wlen is 32 for a 20 fps camera (1.6 s)
    eps = 10**-9
    X = signal
    e, c, f = X.shape            # e = #estimators, c = 3 rgb ch., f = #frames
    w = int(1.6 * kargs['fps'])   # window length

    # stack e times fixed mat P
    P = np.array([[0, 1, -1], [-2, 1, 1]])
    Q = np.stack([P for _ in range(e)], axis=0)

    # Initialize (1)
    H = np.zeros((e, f))
    for n in np.arange(w, f):
        # Start index of sliding window (4)
        m = n - w + 1
        # Temporal normalization (5)
        Cn = X[:, :, m:(n + 1)]
        M = 1.0 / (np.mean(Cn, axis=2)+eps)
        M = np.expand_dims(M, axis=2)  # shape [e, c, w]
        Cn = np.multiply(M, Cn)

        # Projection (6)
        S = np.dot(Q, Cn)
        S = S[0, :, :, :]
        S = np.swapaxes(S, 0, 1)    # remove 3-th dim

        # Tuning (7)
        S1 = S[:, 0, :]
        S2 = S[:, 1, :]
        alpha = np.std(S1, axis=1) / (eps + np.std(S2, axis=1))
        alpha = np.expand_dims(alpha, axis=1)
        Hn = np.add(S1, alpha * S2)
        Hnm = Hn - np.expand_dims(np.mean(Hn, axis=1), axis=1)
        # Overlap-adding (8)
        H[:, m:(n + 1)] = np.add(H[:, m:(n + 1)], Hnm)

    return H


def cupy_POS(signal, **kargs):
    """
    POS method on GPU using Cupy.

    The dictionary parameters are: {'fps':float}.

    Wang, W., den Brinker, A. C., Stuijk, S., & de Haan, G. (2016). Algorithmic principles of remote PPG. IEEE Transactions on Biomedical Engineering, 64(7), 1479-1491. 
    """
    # Run the pos algorithm on the RGB color signal c with sliding window length wlen
    # Recommended value for wlen is 32 for a 20 fps camera (1.6 s)
    eps = 10**-9
    X = signal
    fps = cupy.float32(kargs['fps'])
    e, c, f = X.shape            # e = #estimators, c = 3 rgb ch., f = #frames
    w = int(1.6 * fps)   # window length

    # stack e times fixed mat P
    P = cupy.array([[0, 1, -1], [-2, 1, 1]])
    Q = cupy.stack([P for _ in range(e)], axis=0)

    # Initialize (1)
    H = cupy.zeros((e, f))
    for n in cupy.arange(w, f):
        # Start index of sliding window (4)
        m = n - w + 1
        # Temporal normalization (5)
        Cn = X[:, :, m:(n + 1)]
        M = 1.0 / (cupy.mean(Cn, axis=2)+eps)
        M = cupy.expand_dims(M, axis=2)  # shape [e, c, w]
        Cn = cupy.multiply(M, Cn)

        # Projection (6)
        S = cupy.dot(Q, Cn)
        S = S[0, :, :, :]
        S = cupy.swapaxes(S, 0, 1)    # remove 3-th dim

        # Tuning (7)
        S1 = S[:, 0, :]
        S2 = S[:, 1, :]
        alpha = cupy.std(S1, axis=1) / (eps + cupy.std(S2, axis=1))
        alpha = cupy.expand_dims(alpha, axis=1)
        Hn = cupy.add(S1, alpha * S2)
        Hnm = Hn - cupy.expand_dims(cupy.mean(Hn, axis=1), axis=1)
        # Overlap-adding (8)
        H[:, m:(n + 1)] = cupy.add(H[:, m:(n + 1)], Hnm)

    return H


def cpu_PBV(signal):
    """
    PBV method on CPU using Numpy.

    De Haan, G., & Van Leest, A. (2014). Improved motion robustness of remote-PPG by using the blood volume pulse signature. Physiological measurement, 35(9), 1913.
    """
    sig_mean = np.mean(signal, axis = 2)

    signal_norm_r = signal[:,0,:] / np.expand_dims(sig_mean[:,0],axis=1)
    signal_norm_g = signal[:,1,:] / np.expand_dims(sig_mean[:,1],axis=1)
    signal_norm_b = signal[:,2,:] / np.expand_dims(sig_mean[:,2],axis=1)

    pbv_n = np.array([np.std(signal_norm_r, axis = 1), np.std(signal_norm_g, axis = 1), np.std(signal_norm_b, axis = 1)])
    pbv_d = np.sqrt(np.var(signal_norm_r, axis = 1) + np.var(signal_norm_g, axis = 1) + np.var(signal_norm_b, axis = 1))
    pbv = pbv_n / pbv_d

    C = np.swapaxes(np.array([signal_norm_r, signal_norm_g, signal_norm_b]),0,1)
    Ct =np.swapaxes(np.swapaxes(np.transpose(C),0,2),1,2)
    Q = np.matmul(C, Ct)
    W = np.linalg.solve(Q,np.swapaxes(pbv,0,1))

    A = np.matmul(Ct, np.expand_dims(W,axis = 2))
    B =  np.matmul(np.swapaxes(np.expand_dims(pbv.T,axis=2),1,2),np.expand_dims(W,axis = 2))
    bvp = A / B
    return bvp.squeeze(axis=2)


def cpu_PCA(signal,**kargs):
    """
    PCA method on CPU using Numpy.

    The dictionary parameters are {'component':str}. Where 'component' can be 'second_comp' or 'all_comp'.

    Lewandowska, M., Rumiński, J., Kocejko, T., & Nowak, J. (2011, September). Measuring pulse rate with a webcam—a non-contact method for evaluating cardiac activity. In 2011 federated conference on computer science and information systems (FedCSIS) (pp. 405-410). IEEE.
    """
    bvp = []
    for i in range(signal.shape[0]):
        X = signal[i]
        pca = PCA(n_components=3)
        pca.fit(X)

        # selector
        if kargs['component']=='all_comp':
            bvp.append(pca.components_[0] * pca.explained_variance_[0])
            bvp.append(pca.components_[1] * pca.explained_variance_[1])
        elif kargs['component']=='second_comp':
            bvp.append(pca.components_[1] * pca.explained_variance_[1])
    bvp = np.array(bvp)
    return bvp
    

def cpu_GREEN(signal):
    """
    GREEN method on CPU using Numpy

    Verkruysse, W., Svaasand, L. O., & Nelson, J. S. (2008). Remote plethysmographic imaging using ambient light. Optics express, 16(26), 21434-21445.
    """
    return signal[:,1,:]
    

def cpu_OMIT(signal):
    """
    OMIT method on CPU using Numpy.

    Álvarez Casado, C., Bordallo López, M. (2022). Face2PPG: An unsupervised pipeline for blood volume pulse extraction from faces. arXiv (eprint 2202.04101).
    """

    bvp = []
    for i in range(signal.shape[0]):
        X = signal[i]
        Q, R = np.linalg.qr(X)
        S = Q[:, 0].reshape(1, -1)
        P = np.identity(3) - np.matmul(S.T, S)
        Y = np.dot(P, X)
        bvp.append(Y[1, :])
    bvp = np.array(bvp)
    return bvp

    
def cpu_ICA(signal, **kargs):
    """
    ICA method on CPU using Numpy.

    The dictionary parameters are {'component':str}. Where 'component' can be 'second_comp' or 'all_comp'.

    Poh, M. Z., McDuff, D. J., & Picard, R. W. (2010). Non-contact, automated cardiac pulse measurements using video imaging and blind source separation. Optics express, 18(10), 10762-10774.    
    """
    bvp = []
    for X in signal:
        W = jadeR(X, verbose=False)  
        bvp.append(np.dot(W,X))
    
    # selector
    bvp = np.array(bvp)
    l, c, f = bvp.shape     # l=#landmks c=#3chs, f=#frames
    if kargs['component']=='all_comp':
        bvp = np.reshape(bvp, (l*c, f))  # compact into 2D matrix 
    elif kargs['component']=='second_comp':
        bvp = np.reshape(bvp[:,1,:], (l, f))
    
    # collect
    return bvp

def cpu_SSR(raw_signal,**kargs):
    """
    SSR method on CPU using Numpy.

    'raw_signal' is a float32 ndarray with shape [num_frames, rows, columns, rgb_channels]; it can be obtained by
    using the :py:class:‵pyVHR.extraction.sig_processing.SignalProcessing‵ class ('extract_raw_holistic' method).

    The dictionary parameters are: {'fps':float}.

    Wang, W., Stuijk, S., & De Haan, G. (2015). A novel algorithm for remote photoplethysmography: Spatial subspace rotation. IEEE transactions on biomedical engineering, 63(9), 1974-1984.
    """
    # utils functions #
    def __build_p(τ, k, l, U, Λ):
        """
        builds P
        Parameters
        ----------
        k: int
            The frame index
        l: int
            The temporal stride to use
        U: numpy.ndarray
            The eigenvectors of the c matrix (for all frames up to counter).
        Λ: numpy.ndarray
            The eigenvalues of the c matrix (for all frames up to counter).
        Returns
        -------
        p: numpy.ndarray
            The p signal to add to the pulse.
        """
        # SR'
        SR = np.zeros((3, l), np.float32)  # dim: 3xl
        z = 0

        for t in range(τ, k, 1):  # 6, 7
            a = Λ[0, t]
            b = Λ[1, τ]
            c = Λ[2, τ]
            d = U[:, 0, t].T
            e = U[:, 1, τ]
            f = U[:, 2, τ]
            g = U[:, 1, τ].T
            h = U[:, 2, τ].T
            x1 = a / b
            x2 = a / c
            x3 = np.outer(e, g)
            x4 = np.dot(d, x3)
            x5 = np.outer(f, h)
            x6 = np.dot(d, x5)
            x7 = np.sqrt(x1)
            x8 = np.sqrt(x2)
            x9 = x7 * x4
            x10 = x8 * x6
            x11 = x9 + x10
            SR[:, z] = x11  # 8 | dim: 3
            z += 1

        # build p and add it to the final pulse signal
        s0 = SR[0, :]  # dim: l
        s1 = SR[1, :]  # dim: l
        p = s0 - ((np.std(s0) / np.std(s1)) * s1)  # 10 | dim: l
        p = p - np.mean(p)  # 11
        return p  # dim: l
        
    def __build_correlation_matrix(V):
        # V dim: (W×H)x3
        #V = np.unique(V, axis=0)
        V_T = V.T  # dim: 3x(W×H)
        N = V.shape[0]
        # build the correlation matrix
        C = np.dot(V_T, V)  # dim: 3x3
        C = C / N

        return C

    def __eigs(C):
        """
        get eigenvalues and eigenvectors, sort them.
        Parameters
        ----------
        C: numpy.ndarray
            The RGB values of skin-colored pixels.
        Returns
        -------
        Λ: numpy.ndarray
            The eigenvalues of the correlation matrix
        U: numpy.ndarray
            The (sorted) eigenvectors of the correlation matrix
        """
        # get eigenvectors and sort them according to eigenvalues (largest first)
        L, U = np.linalg.eig(C)  # dim Λ: 3 | dim U: 3x3
        idx = L.argsort()  # dim: 3x1
        idx = idx[::-1]  # dim: 1x3
        L_ = L[idx]  # dim: 3
        U_ = U[:, idx]  # dim: 3x3

        return L_, U_
    # ----------------------------------- #

    fps = int(kargs['fps'])

    raw_sig = raw_signal
    K = len(raw_sig)
    l = int(fps)

    P = np.zeros(K)  # 1 | dim: K
    # store the eigenvalues Λ and the eigenvectors U at each frame
    L = np.zeros((3, K), dtype=np.float32)  # dim: 3xK
    U = np.zeros((3, 3, K), dtype=np.float32)  # dim: 3x3xK

    for k in range(K):
        n_roi = len(raw_sig[k])
        VV = []
        V = raw_sig[k].astype(np.float32)
        idx = V!=0
        idx2 = np.logical_and(np.logical_and(idx[:,:,0], idx[:,:,1]), idx[:,:,2])
        V_skin_only = V[idx2]
        VV.append(V_skin_only)
        
        VV = np.vstack(VV)

        C = __build_correlation_matrix(VV)  #dim: 3x3

        # get: eigenvalues Λ, eigenvectors U
        L[:,k], U[:,:,k] = __eigs(C)  # dim Λ: 3 | dim U: 3x3

        # build p and add it to the pulse signal P
        if k >= l:  # 5
            tau = k - l  # 5
            p = __build_p(tau, k, l, U, L)  # 6, 7, 8, 9, 10, 11 | dim: l
            P[tau:k] += p  # 11

        if np.isnan(np.sum(P)):
            print('NAN')
            print(raw_sig[k])
            
    bvp = P
    bvp = np.expand_dims(bvp,axis=0)
    return bvp
