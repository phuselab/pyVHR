from scipy import signal
import numpy as np
from .base import VHRMethod

class GREEN(VHRMethod):
    methodName = 'GREEN'
    
    def __init__(self, **kwargs):
        super(GREEN, self).__init__(**kwargs)
        
    def apply(self, X):

        bvp = X[1,:].T

        return bvp