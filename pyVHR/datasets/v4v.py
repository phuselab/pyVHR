import csv
import numpy as np
from pyVHR.datasets.dataset import Dataset
from pyVHR.BPM.BPM import BVPsignal


class V4V(Dataset):
    """
    V4V Dataset

    .. V4V dataset structure:
    .. -----------------
    ..     datasetDIR/
    ..     |   |-- SubjDIR1/
    ..     |       |-- vid.avi
    ..     |...
    ..     |   |-- SubjDIRM/
    ..     |       |-- vid.avi
    """
    name = 'V4V'
    signalGT = 'BVP'     # GT signal type
    numLevels = 2        # depth of the filesystem collecting video and BVP files
    numSubjects = 56     # number of subjects
    video_EXT = 'mkv'    # extension of the video files
    frameRate = 25       # vieo frame rate
    VIDEO_SUBSTRING = ''  # substring contained in the filename
    SIG_EXT = 'txt'     # extension of the BVP files
    SIG_SUBSTRING = 'BP'  # substring contained in the filename
    SIG_SampleRate = 1000  # sample rate of the BVP files
    skinThresh = [40, 60]  # thresholds for skin detection


    def readSigfile(self, filename):
        """ 
        Load signal from file.
        
        Returns:
            a :py:class:`pyVHR.BPM.BPM.BVPsignal` object that can be used to extract BPM signal from ground truth BVP signal.
        """
        
        data = np.loadtxt(filename)
        self.SIG_SampleRate = 1000.

        return BVPsignal(data, self.SIG_SampleRate)
