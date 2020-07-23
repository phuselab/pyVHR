import csv
import numpy as np
from pyVHR.datasets.dataset import Dataset
from pyVHR.signals.bvp import BVPsignal

class UBFC1(Dataset):
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
    name = 'UBFC1'
    videodataDIR = '/var/data/VHR/UBFC/DATASET_1/'
    SIGdataDIR = '/var/data/VHR/UBFC/DATASET_1/'
    signalGT = 'BVP'     # GT signal type
    numLevels = 2        # depth of the filesystem collecting video and BVP files
    numSubjects = 8     # number of subjects
    video_EXT = 'avi'    # extension of the video files
    frameRate = 30       # vieo frame rate
    VIDEO_SUBSTRING = ''  # substring contained in the filename
    SIG_EXT = 'xmp'     # extension of the BVP files
    SIG_SUBSTRING = '' # substring contained in the filename
    SIG_SampleRate = 62 # sample rate of the BVP files
    skinThresh = [40,60]  # thresholds for skin detection

    def readSigfile(self, filename):
        """ Load BVP signal.
            Must return a 1-dim (row array) signal
        """
        gtTrace = []
        gtTime = []
        gtHR = []
        with open(filename, 'r') as csvfile:
            xmp = csv.reader(csvfile)
            for row in xmp:
                gtTrace.append(float(row[3]))
                gtTime.append(float(row[0])/1000.)
                gtHR.append(float(row[1]))

        data = np.array(gtTrace)
        time = np.array(gtTime)
        hr = np.array(gtHR)
        self.SIG_SampleRate = np.round(1/np.mean(np.diff(time)))

        '''import matplotlib.pyplot as plt
        plt.plot(hr)
        plt.show()'''

        return BVPsignal(data, self.SIG_SampleRate)
