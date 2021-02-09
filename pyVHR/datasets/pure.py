import json
import numpy as np
import os
from pyVHR.datasets.dataset import Dataset
from pyVHR.signals.bvp import BVPsignal

class PURE(Dataset):
    """
    PURE dataset structure:
    -----------------
        datasetDIR/
        |
        |-- 01-01/
        |---- Image...1.png
        |---- Image.....png
        |---- Image...n.png
        |-- 01-01.json
        |...
        |...
        |-- nn-nn/
        |---- Image...1.png
        |---- Image.....png
        |---- Image...n.png        
        |-- nn-nn.json
        |...
    """
    name = 'PURE'
    signalGT = 'BVP'      # GT signal type
    numLevels = 1         # depth of the filesystem collecting video and BVP files
    numSubjects = 10      # number of subjects
    video_EXT = 'png'     # extension of the video files
    frameRate = 30        # vieo frame rate
    VIDEO_SUBSTRING = '-' # substring contained in the filename
    SIG_EXT = 'json'      # extension of the BVP files
    SIG_SUBSTRING = '-'   # substring contained in the filename
    SIG_SampleRate = 60   # sample rate of the BVP files
    skinThresh = [40,60]  # thresholds for skin detection

    def readSigfile(self, filename):
        """ Load BVP signal.
            Must return a 1-dim (row array) signal
        """
        bvp = []
        with open(filename) as json_file:
            json_data = json.load(json_file)
            for p in json_data['/FullPackage']:
                bvp.append(p['Value']['waveform'])

        data = np.array(bvp)

        return BVPsignal(data, self.SIG_SampleRate)


    def loadFilenames(self):
        """
        Load dataset file names and directories of frames: 
        define vars videoFilenames and BVPFilenames
        """

        # -- loop on the dir struct of the dataset getting directories and filenames
        for root, dirs, files in os.walk(self.videodataDIR):

            for f in files:
                filename = os.path.join(root, f)
                path, name = os.path.split(filename)
                
                # -- select signal
                if name.endswith(self.SIG_EXT) and (name.find(self.SIG_SUBSTRING)>=0):
                    self.sigFilenames.append(filename)
                    self.videoFilenames.append(filename[:-5] + '/file.' + self.video_EXT)

        # -- number of videos
        self.numVideos = len(self.videoFilenames)

