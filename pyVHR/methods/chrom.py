from scipy import signal
import numpy as np
from .base import VHRMethod

class CHROM(VHRMethod):
    """ This method is described in the following paper:
        "Remote heart rate variability for emotional state monitoring"
        by Y. Benezeth, P. Li, R. Macwan, K. Nakamura, R. Gomez, F. Yang
    """
    methodName = 'CHROM'
    
    def __init__(self, **kwargs):
        super(CHROM, self).__init__(**kwargs)
        
    def apply(self, X):

        #self.RGB = self.getMeanRGB()
        #X = signal.detrend(self.RGB.T)
        
        # calculation of new X and Y
        Xcomp = 3*X[0] - 2*X[1]
        Ycomp = (1.5*X[0])+X[1]-(1.5*X[2])

        # standard deviations
        sX = np.std(Xcomp)
        sY = np.std(Ycomp)

        alpha = sX/sY

        # -- rPPG signal
        bvp = Xcomp-alpha*Ycomp

        return bvp