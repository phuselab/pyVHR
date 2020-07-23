import numpy as np
from scipy import signal
from .base import VHRMethod

class LGI(VHRMethod):
    methodName = 'LGI'
    
    def __init__(self, **kwargs):
        super(LGI, self).__init__(**kwargs)
        
    def apply(self, X):

        #M = np.mean(X, axis=1)
        #M = M[:, np.newaxis]
        #Xzero = X - M   # zero mean (row)
        
        U,_,_ = np.linalg.svd(X)

        S = U[:,0].reshape(1,-1) # array 2D shape (1,3)
        P = np.identity(3) - np.matmul(S.T,S)

        Y = np.dot(P,X)
        bvp = Y[1,:]

        return bvp