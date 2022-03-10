import csv
import numpy as np
from pyVHR.datasets.dataset import Dataset
from pyVHR.BPM.BPM import BVPsignal


class VICAR_PPG(Dataset):
    """
    VICAR_PPG Dataset

    .. VICAR_PPG dataset structure:
    .. -----------------
    ..     datasetDIR/
    ..     |   |-- SubjDIR1/
    ..     |       |-- vid.avi
    ..     |...
    ..     |   |-- SubjDIRM/
    ..     |       |-- vid.avi
    """
    name = 'VICAR_PPG'
    signalGT = 'BVP'     # GT signal type
    numLevels = 2        # depth of the filesystem collecting video and BVP files
    numSubjects = 40     # number of subjects
    video_EXT = 'mp4'    # extension of the video files
    frameRate = 60       # vieo frame rate
    VIDEO_SUBSTRING = ''  # substring contained in the filename
    SIG_EXT = 'csv'     # extension of the BVP files
    SIG_SUBSTRING = ''  # substring contained in the filename
    SIG_SampleRate = 1000  # sample rate of the BVP files
    skinThresh = [40, 60]  # thresholds for skin detection


    def readSigfile(self, filename):
        """ 
        Load signal from file.
        
        Returns:
            a :py:class:`pyVHR.BPM.BPM.BVPsignal` object that can be used to extract BPM signal from ground truth BVP signal.
        """
        
        gtTrace = []
        gtTime = []
        with open(filename, 'r') as csvfile:
            dcsv = csv.reader(csvfile)
            for i,row in enumerate(dcsv):
                if i == 0:
                    continue
                gtTime.append(float(row[0])/1000.)
                gtTrace.append(float(row[1]))

        data = np.array(gtTrace)
        time = np.array(gtTime)
        self.SIG_SampleRate = np.round(1/np.mean(np.diff(time)))

        return BVPsignal(data, self.SIG_SampleRate)
