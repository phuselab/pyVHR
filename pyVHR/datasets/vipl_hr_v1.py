import numpy as np
from pyVHR.datasets.dataset import Dataset
from pyVHR.BPM.BPM import BVPsignal
import pandas as pd


class VIPL_HR_V1(Dataset):
    """
    VIPL_HR_V1 Dataset

    .. VIPL_HR_V1 dataset structure:
    .. -----------------
    ..     datasetDIR/
    ..     |
    ..     |-- p1/
    ..          |--v1/
    ..               |--video.avi
    ..               |--wave.csv
    """
    name = 'VIPL_HR_V1'
    signalGT = 'BVP'  # GT signal type
    numLevels = 2  # depth of the filesystem collecting video and BVP files
    numSubjects = 107  # number of subjects
    video_EXT = 'avi'  # extension of the video files
    frameRate = 30  # vieo frame rate
    VIDEO_SUBSTRING = 'video'  # substring contained in the filename
    SIG_EXT = 'csv'  # extension of the BVP files
    SIG_SUBSTRING = 'wave'  # substring contained in the filename
    SIG_SampleRate = 50  # sample rate of the BVP files
    skinThresh = [40, 60]  # thresholds for skin detection

    def readSigfile(self, filename):
        """
        Load signal from file.

        Returns:
            a :py:class:`pyVHR.BPM.BPM.BVPsignal` object that can be used to extract BPM signal from ground truth BVP signal.
        """

        f = open(filename)
        df = pd.read_csv(f)

        data = np.array(df.Wave)  # load the signal

        return BVPsignal(data, self.SIG_SampleRate)
