import configparser
import ast
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from ..datasets.dataset import datasetFactory
from ..methods.base import methodFactory
from ..signals.video import Video
from ..utils.errors import getErrors, printErrors, displayErrors

class TestSuite():
    """ Test suite for a given video dataset and multiple VHR methods"""
    
    def __init__(self, configFilename='default'):
        if configFilename == 'default':
            configFilename = '../pyVHR/analysis/default_test.cfg'
        self.parse_cfg(configFilename)
        
    def start(self, saveResults=True, outFilename=None, verb=0):
        """ Runs the tests as specified in the loaded config file.
            
            verbose degree:
               0 - not verbose
               1 - show the main steps
               2 - display graphic 
               3 - display spectra  
               4 - display errors
               (use also combinations, e.g. verb=21, verb=321)
        """     

        # -- verbose prints
        if '1' in str(verb):
            self.__verbose('a')
            
        # -- dataset & cfg params
        dataset = datasetFactory(self.videodict['dataset'])
        
        # -- catch data (object)
        res = TestResult()
        
        # -- loop on methods 
        for m in self.methods:
            
            # -- loop on videos 
            if self.videoIdx == 'all':
                self.videoIdx = np.arange(0,dataset.numVideos)
            for v in self.videoIdx:
                
                # -- verbose prints
                if '1' in str(verb):
                    print("\n**** Using Method: %s on videoID: %d" % (m,v))
                
                # -- catch data
                res.newDataSerie()
                res.addData('method', m)
                res.addData('dataset', dataset.name)
                res.addData('videoIdx', v)
                
                # -- video object
                videoFilename = dataset.getVideoFilename(v)
                video = Video(videoFilename, verb)
                video.getCroppedFaces(detector=self.videodict['detector'],
                                      extractor=self.videodict['extractor'])
                etime = float(self.videodict['endTime'])
                if etime < 0:
                    self.videodict['endTime'] = str(video.duration-etime)
                # -- catch data
                res.addData('videoFilename', videoFilename)
                
                # -- ground-truth signal
                fname = dataset.getSigFilename(v)
                sigGT = dataset.readSigfile(fname)
                winSizeGT = int(self.methodsdict[m]['winSizeGT'])
                bpmGT, timesGT = sigGT.getBPM(winSizeGT)
                # -- catch data
                res.addData('sigFilename', fname)
                res.addData('bpmGT', sigGT.bpm)
                res.addData('timeGT', sigGT.times)         

                # -- method object
                # load params of m
                self.methodsdict[m]['video'] = video
                self.methodsdict[m]['verb'] = verb
                # merge video parameters dict in method parameters dict before calling method
                self.__merge(self.methodsdict[m], self.videodict)
                method = methodFactory(m, **self.methodsdict[m])
                bpmES, timesES = method.runOffline(**self.methodsdict[m])
                # -- catch data
                res.addData('bpmES', bpmES)
                res.addData('timeES', timesES) 
                
                # -- error metrics
                RMSE, MAE, MAX, PCC = getErrors(bpmES, bpmGT, timesES, timesGT)
                # -- catch data
                res.addData('RMSE', RMSE)
                res.addData('MAE', MAE) 
                res.addData('PCC', PCC)
                res.addData('MAX', MAX)
                res.addDataSerie()
                
                if '1' in str(verb):
                    printErrors(RMSE, MAE, MAX, PCC)
                if '4' in str(verb):
                    displayErrors(bpmES, bpmGT, timesES, timesGT)
                    
        # -- save results on a file
        if saveResults:
            res.saveResults()

        return res
    
    
    def parse_cfg(self, configFilename):
        """ parses the given config file for experiments. """
        
        self.parser = configparser.ConfigParser(inline_comment_prefixes=('#', ';'))
        self.parser.optionxform = str
        if not self.parser.read(configFilename):
            raise FileNotFoundError(configFilename)
            
        # checks 
        assert not self.parser.has_section('DEFAULT'),"ERROR... DEFAULT section is mandatory!"
            
        # load default paramas
        self.defaultdict = dict(self.parser['DEFAULT'].items())
        
        # load video params
        self.videodict = dict(self.parser['VIDEO'].items())
        
        # video idx list extraction
        if self.videodict['videoIdx'] == 'all':
            self.videoIdx = 'all'
        else:
            svid = ast.literal_eval(self.videodict['videoIdx'])
            self.videoIdx = [int(v) for v in svid]

        # load parameters for each methods
        self.methodsdict = {}
        self.methods = ast.literal_eval(self.defaultdict['methods'])
        for x in self.methods:
            self.methodsdict[x] = dict(self.parser[x].items())
        
    def __merge(self, dict1, dict2):
        for key in dict2:
            if key not in dict1:
                dict1[key]= dict2[key]
                
    def __verbose(self, verb):
        if verb == 'a':
            print("** Run the test with the following config:")
            print("      dataset: " + self.videodict['dataset'].upper())
            print("      methods: " + str(self.methods))

            
class TestResult():
    """ Manage the results of a test for a given video dataset and multiple VHR methods"""
    
    def __init__(self, filename=None):

        if filename == None:
            self.dataFrame = pd.DataFrame()
        else:
            self.dataFrame = pd.read_hdf(filename)
        self.dict = None
        
    def addDataSerie(self):
        # -- store serie
        if self.dict != None:
            self.dataFrame = self.dataFrame.append(self.dict, ignore_index=True)
            
    def newDataSerie(self):
        # -- new dict
        D = {}
        D['method'] = ''
        D['dataset'] = ''
        D['videoIdx'] = ''        # video filename
        D['sigFilename'] = ''     # GT signal filename
        D['videoFilename'] = ''   # GT signal filename
        D['EVM'] = False          # True if used, False otherwise
        D['mask'] = ''            # mask used
        D['RMSE'] = ''
        D['MAE'] = ''
        D['PCC'] = ''
        D['MAX'] = ''
        D['telapse'] = ''
        D['bpmGT'] = ''          # GT bpm
        D['bpmES'] = ''
        D['timeGT'] = ''            # GT bpm
        D['timeES'] = ''    
        self.dict = D
    
    def addData(self, key, value):
        self.dict[key] = value
                         
    def saveResults(self, outFilename=None):
        if outFilename == None:
            outFilename = "testResults.h5"
        else:
            self.outFilename = outFilename
        
        # -- save data
        self.dataFrame.to_hdf(outFilename, key='df', mode='w')
