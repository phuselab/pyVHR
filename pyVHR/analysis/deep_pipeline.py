import configparser
import ast
from numpy.lib.arraysetops import isin
import cv2
import numpy
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import plotly.graph_objects as go
from importlib import import_module, util
from pyVHR.datasets.dataset import datasetFactory
from pyVHR.utils.errors import getErrors, printErrors, displayErrors
from pyVHR.extraction.sig_processing import *
from pyVHR.extraction.sig_extraction_methods import *
from pyVHR.extraction.skin_extraction_methods import *
from pyVHR.BVP.BVP import *
from pyVHR.BPM.BPM import *
from pyVHR.BVP.methods import *
from pyVHR.BVP.filters import *
from pyVHR.deepRPPG.mtts_can import *
from pyVHR.extraction.utils import *
import time
from inspect import getmembers, isfunction
import os.path

class DeepPipeline():
    """ 
    This class runs the pyVHR pipeline on a single video or dataset
    """

    def __init__(self):
        pass

    def run_on_dataset(self, configFilename, verb=True):
        """ 
        Runs the tests as specified in the loaded config file.

        Args:
            configFilename:
                - The path to the configuration file
            verb:
                - False - not verbose
                - True - show the main steps
               
               (use also combinations)
        """
        self.configFilename = configFilename
        self.parse_cfg(self.configFilename)
        # -- cfg parser
        parser = configparser.ConfigParser(
            inline_comment_prefixes=('#', ';'))
        parser.optionxform = str
        if not parser.read(self.configFilename):
            raise FileNotFoundError(self.configFilename)

        # -- verbose prints
        if verb:
            self.__verbose('a')

        # -- dataset & cfg params
        if 'path' in self.datasetdict and self.datasetdict['path'] != 'None':
            dataset = datasetFactory(
                self.datasetdict['dataset'], videodataDIR=self.datasetdict['videodataDIR'], BVPdataDIR=self.datasetdict['BVPdataDIR'], path=self.datasetdict['path'])
        else:
            dataset = datasetFactory(
                self.datasetdict['dataset'], videodataDIR=self.datasetdict['videodataDIR'], BVPdataDIR=self.datasetdict['BVPdataDIR'])

        # -- catch data (object)
        res = TestResult()

        # load all the videos
        if self.videoIdx == []:
            self.videoIdx = [int(v)
                             for v in range(len(dataset.videoFilenames))]

        # -- loop on videos
        for v in self.videoIdx:
            # multi-method -> list []

            # -- verbose prints
            if verb:
                print("\n## videoID: %d" % (v))

            # -- ground-truth signal
            try:
                fname = dataset.getSigFilename(v)
                sigGT = dataset.readSigfile(fname)
            except:
                continue
            winSizeGT = int(self.bvpdict['winSize'])
            bpmGT, timesGT = sigGT.getBPM(winSizeGT)

            # -- video file name
            videoFileName = dataset.getVideoFilename(v)
            print(videoFileName)
            fps = get_fps(videoFileName)

            
            sp = SignalProcessing()
            frames = sp.extract_raw(videoFileName)

            #Start chronometer
            #start_time = time.time()

            # -- loop on methods
            for m in self.methods:
                if verb:
                    print("## method: %s" % (str(m)))

                # -- BVP extraction
                if str(m) == 'MTTS_CAN':
                    bvps_pred = MTTS_CAN_deep(frames, fps, verb=0)
                    bvps = BVPsignal(bvps_pred, fps)
                    bvps = bvps.data

                    N = bvps.shape[1]
                    block_idx, timesES = sliding_straded_win_offline(N, winSizeGT, 1, fps)
                    block_signals = []
                    for e in block_idx:
                        st_frame = int(e[0])
                        end_frame = int(e[-1])
                        wind_signal = numpy.copy(bvps[:,st_frame: end_frame+1])
                        block_signals.append(wind_signal)

                    bvps = block_signals
                else:
                    print("Deep Method unsupported!")
                    return

                # POST FILTERING
                postfilter_list = ast.literal_eval(
                    self.bvpdict['post_filtering'])
                if len(postfilter_list) > 0:
                    for f in postfilter_list:
                        if verb:
                            print("  post-filter: %s" % f)
                        fdict = dict(parser[f].items())
                        if fdict['path'] != 'None':
                            # custom path
                            spec = util.spec_from_file_location(
                                fdict['name'], fdict['path'])
                            mod = util.module_from_spec(spec)
                            spec.loader.exec_module(mod)
                            method_to_call = getattr(
                                mod, fdict['name'])
                        else:
                            # package path
                            module = import_module(
                                'pyVHR.BVP.filters')
                            method_to_call = getattr(
                                module, fdict['name'])
                        bvps = apply_filter(
                            bvps, method_to_call, fps=fps, params=ast.literal_eval(fdict['params']))

                # -- BPM extraction
                if self.bpmdict['type'] == 'welch':
                    bpmES = BVP_to_BPM_cuda(bvps, fps, minHz=float(
                        self.bpmdict['minHz']), maxHz=float(self.bpmdict['maxHz']))
                elif self.bpmdict['type'] == 'psd_clustering':
                    bpmES = BVP_to_BPM_PSD_clustering_cuda(bvps, fps, minHz=float(
                        self.bpmdict['minHz']), maxHz=float(self.bpmdict['maxHz']))
                   
                # median BPM from multiple estimators BPM
                median_bpmES, mad_bpmES = multi_est_BPM_median(bpmES)

                #end_time = time.time()
                #time_elapsed = [end_time - start_time]

                # -- error metrics
                RMSE, MAE, MAX, PCC, CCC, SNR = getErrors(bvps, fps,
                    np.expand_dims(median_bpmES, axis=0), bpmGT, timesES, timesGT)

                # -- save results
                res.newDataSerie()
                res.addData('dataset', str(self.datasetdict['dataset']))
                res.addData('method', str(m))
                res.addData('videoIdx', v)
                res.addData('RMSE', RMSE)
                res.addData('MAE', MAE)
                res.addData('MAX', MAX)
                res.addData('PCC', PCC)
                res.addData('CCC', CCC)
                res.addData('SNR', SNR)
                res.addData('bpmGT', bpmGT)
                res.addData('bpmES', median_bpmES)
                res.addData('bpmES_mad', mad_bpmES)
                res.addData('timeGT', timesGT)
                res.addData('timeES', timesES)
                res.addData('videoFilename', videoFileName)
                res.addDataSerie()

                if verb:
                    printErrors(RMSE, MAE, MAX, PCC, CCC, SNR)

        return res

    def parse_cfg(self, configFilename):
        """ parses the given configuration file for loading the test's parameters.
        
        Args:
            configFilename: configuation file (.cfg) name of path .

        """

        self.parser = configparser.ConfigParser(
            inline_comment_prefixes=('#', ';'))
        self.parser.optionxform = str
        if not self.parser.read(configFilename):
            raise FileNotFoundError(configFilename)

        # load paramas
        self.datasetdict = dict(self.parser['DATASET'].items())
        self.bvpdict = dict(self.parser['BVP'].items())
        self.bpmdict = dict(self.parser['BPM'].items())

        # video idx list extraction
        if isinstance(ast.literal_eval(self.datasetdict['videoIdx']), list):
            self.videoIdx = [int(v) for v in ast.literal_eval(
                self.datasetdict['videoIdx'])]

        # method list extraction
        if isinstance(ast.literal_eval(self.bvpdict['methods']), list):
            self.methods = ast.literal_eval(
                    self.bvpdict['methods'])

    def __merge(self, dict1, dict2):
        for key in dict2:
            if key not in dict1:
                dict1[key] = dict2[key]

    def __verbose(self, verb):
        if verb == 'a':
            print("** Run the test with the following config:")
            print("      dataset: " + self.datasetdict['dataset'].upper())
            print("      methods: " + str(self.methods))


class TestResult():
    """ 
    This class is used by :py:class:`pyVHR.analysis.newsuite.NewSuite` to manage the results
    of a test for a given video dataset on multiple rPPG methods
    """

    def __init__(self, filename=None):

        if filename == None:
            self.dataFrame = pd.DataFrame()
        else:
            self.dataFrame = pd.read_hdf(filename)
        self.dict = None

    def addDataSerie(self):
        # -- store serie
        if self.dict != None:
            self.dataFrame = self.dataFrame.append(
                self.dict, ignore_index=True)

    def newDataSerie(self):
        # -- new dict
        D = {}
        D['method'] = ''
        D['dataset'] = ''
        D['videoIdx'] = ''        # video filename
        D['sigFilename'] = ''     # GT signal filename
        D['videoFilename'] = ''   # GT signal filename
        D['RMSE'] = ''
        D['MAE'] = ''
        D['PCC'] = ''
        D['CCC'] = ''
        D['SNR'] = ''
        D['MAX'] = ''
        D['bpmGT'] = ''          # GT bpm
        D['bpmES'] = ''
        D['bpmES_mad'] = ''
        D['timeGT'] = ''            # GT bpm
        D['timeES'] = ''
        D['TIME_REQUIREMENT'] = ''
        self.dict = D

    def addData(self, key, value):
        self.dict[key] = value

    def saveResults(self, outFilename=None):
        """
        Save the test results in a HDF5 library that can be opened using pandas.
        You can analyze the results using :py:class:`pyVHR.analysis.stats.StatAnalysis`
        """
        if outFilename == None:
            outFilename = "testResults.h5"
        else:
            self.outFilename = outFilename

        # -- save data
        self.dataFrame.to_hdf(outFilename, key='df', mode='w')
