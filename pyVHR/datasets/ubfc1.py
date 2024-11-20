import csv
import numpy as np
from pyVHR.datasets.dataset import Dataset
from pyVHR.BPM.BPM import BVPsignal


class UBFC1(Dataset):
    """
    UBFC1 Dataset

    .. UBFC dataset structure:
    .. -----------------
    ..     datasetDIR/
    ..     |   |-- SubjDIR1/
    ..     |       |-- vid.avi
    ..     |...
    ..     |   |-- SubjDIRM/
    ..     |       |-- vid.avi
    """
    name = 'UBFC1'
    signalGT = 'BVP'     # GT signal type
    numLevels = 2        # depth of the filesystem collecting video and BVP files
    numSubjects = 8     # number of subjects
    video_EXT = 'avi'    # extension of the video files
    frameRate = 30       # video frame rate
    VIDEO_SUBSTRING = ''  # substring contained in the filename
    SIG_EXT = 'xmp'     # extension of the BVP files
    SIG_SUBSTRING = ''  # substring contained in the filename
    SIG_SampleRate = 62  # sample rate of the BVP files
    skinThresh = [40, 60]  # thresholds for skin detection

    def readSigfile(self, filename):
        """ 
        Load signal from file.
        
        Returns:
            a :py:class:`pyVHR.BPM.BPM.BVPsignal` object that can be used to extract BPM signal from ground truth BVP signal.
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
