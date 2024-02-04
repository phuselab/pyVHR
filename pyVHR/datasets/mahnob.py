import numpy as np
import pybdf
from biosppy.signals import ecg
from pyVHR.datasets.dataset import Dataset
from pyVHR.utils.ecg import ECGsignal

"""
In order to use this module you need a Fortran Compiler.

For Linux you can use:
    sudo apt-get install gfortran

"""


class MAHNOB(Dataset):
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
    name = 'MAHNOB'
    signalGT = 'ECG'  # GT signal type
    numLevels = 2  # depth of the filesystem collecting video and BVP files
    numSubjects = 40  # number of subjects
    video_EXT = 'avi'  # extension of the video files
    frameRate = 61  # video frame rate
    VIDEO_SUBSTRING = 'Section'  # substring contained in the filename
    SIG_EXT = 'bdf'  # extension of the ECG files
    SIG_SUBSTRING = 'emotion'  # substring contained in the filename
    SIG_SampleRate = 256  # sample rate of the ECG files

    def readSigfile(self, filename):
        """ Load ECG signal.

            Returns:
                a pyVHR.utils.ecg.ECGsignal object that can be used to extract ground truth BPM signal.
            
        """
        bdfRec = pybdf.bdfRecording(filename)
        rec = bdfRec.getData(channels=[33])
        self.SIG_SampleRate = bdfRec.sampRate[33]
        data = np.array(rec['data'][0])[rec['eventTable']['idx'][2]:]
        return ECGsignal(data, self.SIG_SampleRate)
