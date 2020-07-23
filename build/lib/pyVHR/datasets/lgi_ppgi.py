import xml.etree.ElementTree as ET
import numpy as np
from os import path
from pyVHR.datasets.dataset import Dataset
from pyVHR.signals.bvp import BVPsignal

class LGI_PPGI(Dataset):
    """
    Alex dataset structure:
    -----------------
        datasetDIR/
        |
        |-- vidDIR1/
        |   |-- videoFile1.avi
        |
        |...
        |
        |-- vidDIRM/
            |-- videoFile1.avi
    """
    name = 'LGI_PPGI'
    videodataDIR = '/var/data/VHR/LGI-PPGI/'
    BVPdataDIR = '/var/data/VHR/LGI-PPGI/'
    signalGT = 'BVP'          # GT signal type
    numLevels = 2             # depth of the filesystem collecting video and BVP files
    numSubjects = 4           # number of subjects
    video_EXT = 'avi'         # extension of the video files
    frameRate = 25            # vieo frame rate
    VIDEO_SUBSTRING = 'cv_camera'  # substring contained in the filename
    SIG_EXT = 'xml'           # extension of the BVP files
    SIG_SUBSTRING = 'cms50'   # substring contained in the filename
    SIG_SampleRate = 60       # sample rate of the BVP files

    def readSigfile(self, filename):
        """
            Load BVP signal. Must return a 1-dim (row array) signal
        """

        tree = ET.parse(filename)
        # get all bvp elements and their values
        bvp_elements = tree.findall('.//*/value2')
        bvp = [int(item.text) for item in bvp_elements]
        
        n_bvp_samples = len(bvp)
        last_bvp_time = int((n_bvp_samples*1000)/self.SIG_SampleRate)

        vid_xml_filename = path.join(path.dirname(filename), 'cv_camera_sensor_timer_stream_handler.xml')
        tree = ET.parse(vid_xml_filename)
        
        root = tree.getroot()
        last_vid_time = int(float(root[-1].find('value1').text))

        diff = ((last_bvp_time - last_vid_time)/1000)

        assert diff >= 0, 'Unusable data.'

        print("Skipping %.2f seconds..." % diff)

        diff_samples = round(diff*self.SIG_SampleRate)

        data = np.array(bvp[diff_samples:])

        return BVPsignal(data, self.SIG_SampleRate)
