from abc import ABCMeta, abstractmethod
import os
from importlib import import_module, util

# def datasetFactory(datasetName, *args, **kwargs):


def datasetFactory(datasetName, videodataDIR, BVPdataDIR, path=None):
    """
    This method is used for creating a new istance of a dataset Class (that
    innherit :py:class:`pyVHR.datasets.dataset.Dataset` ).

    Args:
        datasetName (str): name of the dataset Class.
        videodataDIR (str): path of the video data directory.
        BVPdataDIR (str): path of the ground truth BVP data directory.
    """
    try:
        if path == None:
            moduleName = datasetName.lower()
            className = datasetName.upper()
            datasetModule = import_module(
                'pyVHR.datasets.' + moduleName)
            classOBJ = getattr(datasetModule, className)
            obj = classOBJ(videodataDIR, BVPdataDIR)
        else:
            moduleName = datasetName.lower()
            className = datasetName.upper()
            relpath = str(path) + str(moduleName) + '.py'
            spec = util.spec_from_file_location(
                moduleName, relpath)
            mod = util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            classOBJ = getattr(mod, className)
            obj = classOBJ(videodataDIR, BVPdataDIR)

    except (AttributeError, ModuleNotFoundError):
        raise ImportError(
            '{} is not part of pyVHR dataset collection!'.format(datasetName))

    return obj


class Dataset(metaclass=ABCMeta):
    """
    This is the abstract class used for creating a new Dataset Class. 
    """

    def __init__(self, videodataDIR=None, BVPdataDIR=None):
        """
        Args:
            videodataDIR (str): path of the video data directory.
            BVPdataDIR (str): path of the ground truth BVP data directory.
        """
        # -- load filenames
        self.videoFilenames = []  # list of all video filenames
        self.sigFilenames = []    # list of all Sig filenames
        self.numVideos = 0        # num of videos in the dataset
        self.videodataDIR = videodataDIR
        self.BVPdataDIR = BVPdataDIR
        self.loadFilenames()

    def loadFilenames(self):
        """Load dataset file names: define vars videoFilenames and BVPFilenames."""

        # -- loop on the dir struct of the dataset getting filenames
        for root, dirs, files in os.walk(self.videodataDIR):
            for f in files:
                filename = os.path.join(root, f)
                path, name = os.path.split(filename)

                # -- select video
                if filename.endswith(self.video_EXT) and (name.find(self.VIDEO_SUBSTRING) >= 0):
                    self.videoFilenames.append(filename)

                # -- select signal
                if filename.endswith(self.SIG_EXT) and (name.find(self.SIG_SUBSTRING) >= 0):
                    self.sigFilenames.append(filename)

        # -- number of videos
        self.numVideos = len(self.videoFilenames)

    def getVideoFilename(self, videoIdx=0):
        """Get video file name given the progressive index."""
        return self.videoFilenames[videoIdx]

    def getSigFilename(self, videoIdx=0):
        """Get Signal file name given the progressive index."""
        return self.sigFilenames[videoIdx]

    @abstractmethod
    def readSigfile(self, filename):
        """ 
        Load signal from file.
        
        Returns:
            a :py:class:`pyVHR.BPM.BPM.BVPsignal` object that can be used to extract BPM signal from ground truth BVP signal.
        """
        pass
