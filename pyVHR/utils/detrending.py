import numpy as np
import scipy.sparse

def detrend(X, detLambda=10):
    # Smoothness prior approach as in the paper appendix:
    # "An advanced detrending method with application to HRV analysis"
    # by Tarvainen, Ranta-aho and Karjaalainen
    t = X.shape[0]
    l = t/detLambda #lambda
    I = np.identity(t)
    D2 = scipy.sparse.diags([1, -2, 1], [0,1,2],shape=(t-2,t)).toarray() # this works better than spdiags in python
    detrendedX = (I-np.linalg.inv(I+l**2*(np.transpose(D2).dot(D2)))).dot(X)
    return detrendedX
