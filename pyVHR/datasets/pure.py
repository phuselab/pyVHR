import json
import numpy as np
import os
from pyVHR.datasets.dataset import Dataset
from pyVHR.BPM.BPM import BVPsignal
import glob
import re
import cv2

class PURE(Dataset):
    """
    PURE Dataset

    .. PURE dataset structure:
    .. -----------------
    ..     datasetDIR/
    ..     |
    ..     |-- 01-01/
    ..     |---- Image...1.png
    ..     |---- Image.....png
    ..     |---- Image...n.png
    ..     |-- 01-01.json
    ..     |...
    ..     |...
    ..     |-- nn-nn/
    ..     |---- Image...1.png
    ..     |---- Image.....png
    ..     |---- Image...n.png        
    ..     |-- nn-nn.json
    ..     |...
    """
    name = 'PURE'
    signalGT = 'BVP'      # GT signal type
    numLevels = 1         # depth of the filesystem collecting video and BVP files
    numSubjects = 10      # number of subjects
    video_EXT = 'avi'     # extension of the video files ### IMPORTANT NOTE: pure datasets consists of a sequence of png files, we converted them in a video avi files
    frameRate = 30.0        # vieo frame rate
    VIDEO_SUBSTRING = '-'  # substring contained in the filename
    SIG_EXT = 'json'      # extension of the BVP files
    SIG_SUBSTRING = '-'   # substring contained in the filename
    SIG_SampleRate = 60   # sample rate of the BVP files
    skinThresh = [40, 60]  # thresholds for skin detection

    def readSigfile(self, filename):
        """ 
        Load signal from file.
        
        Returns:
            a :py:class:`pyVHR.BPM.BPM.BVPsignal` object that can be used to extract BPM signal from ground truth BVP signal.
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

        if self.name not in self.videodataDIR:
            if self.videodataDIR[-1]=='/':
                self.videodataDIR += self.name
            else:
                self.videodataDIR = self.videodataDIR + '/' + self.name

        print(self.videodataDIR)
        #check if video files exist or create them from images
        for root, dirs, files in os.walk(self.videodataDIR):
            for d in dirs:
                dirname = os.path.join(root, d)
                if not glob.glob(dirname +'/*.avi'):
                    #create videofile from images
                    frames = self.__loadFrames(dirname)
                    fps = self.frameRate
                    width = frames[0].shape[1]
                    height = frames[0].shape[0]
                    #fourcc = cv2.VideoWriter_fourcc(*'MPNG')
                    writer = cv2.VideoWriter(dirname + '/' + d + '.avi', 0, fps, (width, height))
                    for frame in frames:
                        writer.write(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                    writer.release()
        # -- loop on the dir struct of the dataset getting directories and filenames
        for root, dirs, files in os.walk(self.videodataDIR):

            for f in files:
                filename = os.path.join(root, f)
                path, name = os.path.split(filename)

                # -- select signal
                if name.endswith(self.SIG_EXT) and (name.find(self.SIG_SUBSTRING) >= 0):
                    self.sigFilenames.append(filename)
                    self.videoFilenames.append(filename[:-5] + '/' + name[:-5] + '.' + self.video_EXT)

        # -- number of videos
        self.numVideos = len(self.videoFilenames)

    def __sort_nicely(self, l): 
        """ Sort the given list in the way that humans expect. 
        """ 
        convert = lambda text: int(text) if text.isdigit() else text 
        alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ] 
        l.sort( key=alphanum_key )
        return l

    def __loadFrames(self, directorypath):
        # -- get filenames within dir
        f_names = self.__sort_nicely(os.listdir(directorypath))
        frames = []
        for n in range(len(f_names)):
            filename = os.path.join(directorypath, f_names[n])
            frames.append(cv2.imread(filename)[:, :, ::-1])
        
        frames = np.array(frames)
        return frames
