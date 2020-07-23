import numpy as np
from scipy import signal
from .base import VHRMethod

class POS(VHRMethod):
    """  
        POS algorithm described in "Algorithmic Principles of Remote PPG"
        (https://ieeexplore.ieee.org/document/7565547 )
        Numbers in brackets refer to the line numbers in the "Algorithm 1" of the paper
    """
    
    methodName = 'POS'
    projection = np.array([[0, 1, -1], [-2, 1, 1]])

    def __init__(self, **kwargs):
        super(POS, self).__init__(**kwargs)

    def apply(self, X):
        # Run the pos algorithm on the RGB color signal c with sliding window length wlen
        # Recommended value for wlen is 32 for a 20 fps camera (1.6 s)
        
        wlen = int(1.6*self.video.frameRate)

        # Initialize (1)
        h = np.zeros(X.shape[1])
        for n in range(X.shape[1]):
            # Start index of sliding window (4)
            m = n - wlen + 1
            if m >= 0:
                # Temporal normalization (5)
                cn = X[:, m:(n+1)]
                cn = np.dot(self.__get_normalization_matrix(cn), cn)
                # Projection (6)
                s = np.dot(self.projection, cn)
                # Tuning (7)
                hn = np.add(s[0, :], np.std(s[0, :])/np.std(s[1, :])*s[1, :])
                # Overlap-adding (8)
                h[m:(n+1)] = np.add(h[m:(n+1)], hn - np.mean(hn))
        return h


    def __get_normalization_matrix(self, x):
        # Compute a diagonal matrix n such that the mean of n*x is a vector of ones
        d = 0 if (len(x.shape) < 2) else 1
        m = np.mean(x, d)
        n = np.array([[1/m[i] if i == j and m[i] else 0 for i in range(len(m))] for j in range(len(m))])
        return n
