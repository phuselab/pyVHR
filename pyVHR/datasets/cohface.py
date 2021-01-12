import h5py
import numpy as np
from pyVHR.datasets.dataset import Dataset
from pyVHR.signals.bvp import BVPsignal

class COHFACE(Dataset):
    """
    Cohface dataset structure:
    -----------------
        datasetDIR/
        |
        |-- subjDIR_1/
        |   |-- vidDIR1/
        |       |-- videoFile1.avi
        |       |-- ...
        |       |-- videoFileN.avi
        |...
        |   |-- vidDIRM/
        |       |-- videoFile1.avi
        |       |-- ...
        |       |-- videoFileM.avi
        |...
        |-- subjDIR_n/
        |...
    """
    name = 'COHFACE'
    signalGT = 'BVP'         # GT signal type
    numLevels = 2            # depth of the filesystem collecting video and BVP files
    numSubjects = 40         # number of subjects
    video_EXT = 'avi'        # extension of the video files
    frameRate = 20           # vieo frame rate
    VIDEO_SUBSTRING = 'data' # substring contained in the filename
    SIG_EXT = 'hdf5'         # extension of the BVP files
    SIG_SUBSTRING = 'data'   # substring contained in the filename
    SIG_SampleRate = 256     # sample rate of the BVP files
    skinThresh = [40,60]     # thresholds for skin detection

    def readSigfile(self, filename):
        """ Load BVP signal.
            Must return a 1-dim (row array) signal
        """

        f = h5py.File(filename, 'r')
        data = np.array(f['pulse'])        # load the signal
        data = data.reshape(1,len(data))   # monodimentional signal

        return BVPsignal(data, self.SIG_SampleRate)
