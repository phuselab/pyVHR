from abc import ABCMeta, abstractmethod
import os
from importlib import import_module

def datasetFactory(datasetName, *args, **kwargs):
    try:
        moduleName = datasetName.lower()
        className = datasetName.upper()
        datasetModule = import_module('pyVHR.datasets.' + moduleName) #, package='pyVHR')
        classOBJ = getattr(datasetModule, className)
        obj = classOBJ(*args, **kwargs)

    except (AttributeError, ModuleNotFoundError):
        raise ImportError('{} is not part of pyVHR dataset collection!'.format(datasetName))

    return obj

class Dataset(metaclass=ABCMeta):
    """
    Manage datasets (parent class for new datasets)
    """
    def __init__(self):
        # -- load filenames
        self.videoFilenames = []  # list of all video filenames
        self.sigFilenames = []    # list of all Sig filenames
        self.numVideos = 0        # num of videos in the dataset
        self.loadFilenames()

    def loadFilenames(self):
        """Load dataset file names: define vars videoFilenames and BVPFilenames"""

        # -- loop on the dir struct of the dataset getting filenames
        for root, dirs, files in os.walk(self.videodataDIR):
            for f in files:
                filename = os.path.join(root, f)
                path, name = os.path.split(filename)

                # -- select video
                if filename.endswith(self.video_EXT) and (name.find(self.VIDEO_SUBSTRING)>=0):
                    self.videoFilenames.append(filename)
                   
                # -- select signal
                if filename.endswith(self.SIG_EXT) and (name.find(self.SIG_SUBSTRING)>=0):
                    self.sigFilenames.append(filename)

        # -- number of videos
        self.numVideos = len(self.videoFilenames)

    def getVideoFilename(self, videoIdx=0):
        """Get video filename given the progressive index"""
        return self.videoFilenames[videoIdx]

    def getSigFilename(self, videoIdx=0):
        """Get Signal filename given the progressive index"""
        return self.sigFilenames[videoIdx]

    @abstractmethod
    def readSigfile(self, filename):
        """ Load signal from file.
            Return a BVPsignal/ECGsignal object.
        """
        pass
