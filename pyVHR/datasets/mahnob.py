
import numpy as np
import pybdf
from biosppy.signals import ecg
from pyVHR.datasets.dataset import Dataset
from pyVHR.signals.ecg import ECGsignal

class MAHNOB(Dataset):
    """
    Mahnob dataset structure:
    -----------------
        datasetDIR/
        |
        ||-- vidDIR1/
        |   |-- videoFile.avi
        |   |-- physioFile.bdf
        |...
        |...
    """
    name = 'MAHNOB'
    videodataDIR = '/var/data/VHR/Mahnob/'
    SIGdataDIR = '/var/data/VHR/Mahnob/'
    signalGT = 'ECG'     # GT signal type
    numLevels = 2        # depth of the filesystem collecting video and BVP files
    numSubjects = 40     # number of subjects
    video_EXT = 'avi'    # extension of the video files
    frameRate = 20       # video frame rate
    VIDEO_SUBSTRING = 'Section'  # substring contained in the filename
    SIG_EXT = 'bdf'     # extension of the ECG files
    SIG_SUBSTRING = 'emotion' # substring contained in the filename
    SIG_SampleRate = 256 # sample rate of the ECG files

    def readSigfile(self, filename):
        """ Load ECG signal.
            Return a 2-dim signal (t, bmp(t))
        """
        bdfRec = pybdf.bdfRecording(filename)
        rec = bdfRec.getData(channels=[33])
        self.SIG_SampleRate = bdfRec.sampRate[33]
        data = np.array(rec['data'][0])[rec['eventTable']['idx'][2]:]
        return ECGsignal(data, self.SIG_SampleRate)
        
