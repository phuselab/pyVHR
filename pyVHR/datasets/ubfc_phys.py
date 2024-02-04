import csv
import numpy as np
from pyVHR.datasets.dataset import Dataset
from pyVHR.BPM.BPM import BVPsignal


class UBFC_PHYS(Dataset):
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
    name = 'UBFC_PHYS'
    signalGT = 'BVP'     # GT signal type
    numLevels = 2        # depth of the filesystem collecting video and BVP files
    numSubjects = 56     # number of subjects
    video_EXT = 'avi'    # extension of the video files
    frameRate = 35       # video frame rate
    VIDEO_SUBSTRING = ''  # substring contained in the filename
    SIG_EXT = 'csv'     # extension of the BVP files
    SIG_SUBSTRING = 'bvp'  # substring contained in the filename
    SIG_SampleRate = 64  # sample rate of the BVP files
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
            d = csv.reader(csvfile)
            for row in d:
                gtTrace.append(float(row[0]))

        data = np.array(gtTrace)
        self.SIG_SampleRate = 64

        #import code; code.interact(local=locals())

        '''import matplotlib.pyplot as plt
        plt.plot(data)
        plt.show()'''

        return BVPsignal(data, self.SIG_SampleRate)
