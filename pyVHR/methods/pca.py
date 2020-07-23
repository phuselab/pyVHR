from sklearn import decomposition
from numpy import vstack
from .base import VHRMethod

class PCA(VHRMethod):
    methodName = 'PCA'
    
    def __init__(self, **kwargs):
        super(PCA, self).__init__(**kwargs)

    def apply(self, X):
        
        # TODO: preproc
        #X = self.__preprocess(X.T)

        bvp = decomposition.PCA(n_components=3).fit_transform(X.T).T

        return bvp


    def __preprocess(self, X):
        
        R = X[:,0].copy()
        G = X[:,1].copy()
        B = X[:,2].copy()

        # -- BP pre-filtering of RGB channels
        minHz = BVPsignal.minHz
        maxHz = BVPsignal.maxHz
        fs = self.video.frameRate

        # -- filter
        filteredR = BPfilter(R, minHz, maxHz, fs)
        filteredG = BPfilter(G, minHz, maxHz, fs)
        filteredB = BPfilter(B, minHz, maxHz, fs)

        X = vstack([filteredR, filteredG, filteredB])

        return X