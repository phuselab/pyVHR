import numpy as np
from scipy import signal
from .base import VHRMethod

class PBV(VHRMethod):
    methodName = 'PBV'
    
    def __init__(self, **kwargs):
        super(PBV, self).__init__(**kwargs)
        
    def apply(self, X):
        
        r_mean = X[0,:]/np.mean(X[0,:])
        g_mean = X[1,:]/np.mean(X[1,:])
        b_mean = X[2,:]/np.mean(X[2,:])

        pbv_n = np.array([np.std(r_mean), np.std(g_mean), np.std(b_mean)])
        pbv_d = np.sqrt(np.var(r_mean) + np.var(g_mean) + np.var(b_mean))
        pbv = pbv_n / pbv_d

        C = np.array([r_mean, g_mean, b_mean])
        Q = np.matmul(C ,np.transpose(C))
        W = np.linalg.solve(Q,pbv)

        bvp = np.matmul(C.T,W)/(np.matmul(pbv.T,W)) 

        return bvp