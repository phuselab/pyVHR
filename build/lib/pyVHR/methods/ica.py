import numpy as np
from scipy import signal
from .utils.jade import jadeR
from .base import VHRMethod

class ICA(VHRMethod):
    methodName = 'ICA'
    
    def __init__(self, **kwargs):
        self.tech = kwargs['ICAmethod']  
        super(ICA, self).__init__(**kwargs)

    def apply(self, X):
        """ ICA method """
        
        # -- JADE (ICA)
        if self.tech == 'jade':
            W = self.__jade(X)
        elif 'fastICA':
            W = self.__fastICA(X)
            
        bvp = np.dot(W,X) # 3-dim signal!!
        
        return bvp


    def __jade(self, X):
        W = np.asarray(jadeR(X, 3, False))
        return W

    def __fastICA(self, X):
        from sklearn.decomposition import FastICA, PCA
        from numpy.linalg import inv, eig

        # -- PCA
        pca = PCA(n_components=3)
        Y = pca.fit_transform(X)

        # -- ICA
        ica = FastICA(n_components=3, max_iter=2000)
        S = ica.fit_transform(Y)

        return S.T        