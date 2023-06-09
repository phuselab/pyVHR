import numpy as np
from pyVHR.datasets.dataset import Dataset
from pyVHR.utils.ecg import ECGsignal
import csv
import os

"""
In order to use this module you need a Fortran Compiler.

For Linux you can use:
    sudo apt-get install gfortran

"""

class ECG_FITNESS(Dataset):
    """
    Mahnob Dataset

    .. Mahnob dataset structure:
    .. -----------------
    ..     datasetDIR/
    ..     |
    ..     ||-- vidDIR1/
    ..     |   |-- videoFile.avi
    ..     |   |-- physioFile.bdf
    ..     |...
    ..     |...
    """
    name = 'ECG_FITNESS'
    signalGT = 'ECG'     # GT signal type
    numLevels = 2        # depth of the filesystem collecting video and BVP files
    numSubjects = 16     # number of subjects
    video_EXT = 'avi'    # extension of the video files
    frameRate = 20       # video frame rate
    VIDEO_SUBSTRING = ''  # substring contained in the filename
    SIG_EXT = 'csv'     # extension of the ECG files
    SIG_SUBSTRING = 'viatom'  # substring contained in the filename
    SIG_SampleRate = 120  # sample rate of the ECG files

    def readSigfile(self, filename):
        """ Load ECG signal.

            Returns:
                a pyVHR.utils.ecg.ECGsignal object that can be used to extract ground truth BPM signal.
            
        """
        gtTime = []
        ecgTrace = []
        ecgHR = []
        with open(filename, 'r') as csvfile:
            d = csv.reader(csvfile)
            for ir,row in enumerate(d):
                if ir == 0:
                    continue
                gtTime.append(float(row[0]))
                ecgTrace.append(float(row[1]))
                ecgHR.append(float(row[2]))
        
        gtTime = np.unique(np.array(gtTime))
        ecg_sr = np.round(1/(np.mean(np.diff(gtTime))/1000.)) * 5
        self.SIG_SampleRate = ecg_sr #120
        data = np.array(ecgTrace)

        path, _ = os.path.split(filename)

        with open(path+'/c920.csv', 'r') as csvfile:
            d = list(csv.reader(csvfile))
            start_ecg = int(d[0][1])
            end_ecg = int(d[-1][1]) + 1

        data = data[start_ecg:end_ecg]

        return ECGsignal(data, self.SIG_SampleRate)
        
