import csv
import numpy as np
from pyVHR.datasets.dataset import Dataset
from pyVHR.signals.bvp import BVPsignal

class UBFC2(Dataset):
    """
    UBFC dataset structure:
    -----------------
        datasetDIR/
        |   |-- SubjDIR1/
        |       |-- vid.avi
        |...
        |   |-- SubjDIRM/
        |       |-- vid.avi
    """
    name = 'UBFC2'
    signalGT = 'BVP'     # GT signal type
    numLevels = 2        # depth of the filesystem collecting video and BVP files
    numSubjects = 26     # number of subjects
    video_EXT = 'avi'    # extension of the video files
    frameRate = 30       # vieo frame rate
    VIDEO_SUBSTRING = ''  # substring contained in the filename
    SIG_EXT = 'txt'     # extension of the BVP files
    SIG_SUBSTRING = '' # substring contained in the filename
    SIG_SampleRate = 30 # sample rate of the BVP files
    skinThresh = [40,60]  # thresholds for skin detection

    def readSigfile(self, filename):
        """ Load BVP signal.
            Must return a 1-dim (row array) signal
        """
        gtTrace = []
        gtTime = []
        gtHR = []
        with open(filename, 'r') as f:
            x = f.readlines()
        
        s = x[0].split(' ')
        s = list(filter(lambda a: a != '', s))
        gtTrace = np.array(s).astype(np.float64)

        t = x[2].split(' ')
        t = list(filter(lambda a: a != '', t))
        gtTime = np.array(t).astype(np.float64)
        
        hr = x[1].split(' ')
        hr = list(filter(lambda a: a != '', hr))
        gtHR = np.array(hr).astype(np.float64)

        data = np.array(gtTrace)
        time = np.array(gtTime)
        self.SIG_SampleRate = np.round(1/np.mean(np.diff(time)))

        return BVPsignal(data, self.SIG_SampleRate)
