import configparser
import ast
from numpy.lib.arraysetops import isin
import cv2
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


class MultiMethodSuite():
    """ 
    This class performs tests on a video dataset using multiple rPPG methods.
    The suite uses these methods individually to calculate BVP and estimate BPM. 
    After this all the BVP signals of the methods are combined into a single n-estimator signal, 
    which is used to estimate a last BPM signal. 
    Using multiple methods together allows you to have many estimators, 
    so it can be useful to have a more precise BPM estimate.
    You can customize all the parameters using a configuration file (.cfg); in the
    analysis module you can find an example of cfg file named "multi_method_cfg.cfg".
    """

    def __init__(self, configFilename='default'):
        self.configFilename = configFilename
        if configFilename == 'default':
            self.configFilename = '../pyVHR/analysis/newcfg.cfg'
        self.parse_cfg(self.configFilename)

    def start(self, verb=0):
        """ 
        Runs the tests as specified in the loaded config file.

        Args:
            verb:
               - 0 - not verbose
               - 1 - show the main steps
               
               (use also combinations)
        """

        # -- cfg parser
        parser = configparser.ConfigParser(
            inline_comment_prefixes=('#', ';'))
        parser.optionxform = str
        if not parser.read(self.configFilename):
            raise FileNotFoundError(self.configFilename)

        # -- verbose prints
        if '1' in str(verb):
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

        # -- SIG processing
        sig_processing = SignalProcessing()
        if bool(self.sigdict['cuda']):
            sig_processing.display_cuda_device()
            sig_processing.choose_cuda_device(int(self.sigdict['cuda_device']))
        # set skin extractor
        target_device = 'GPU' if bool(self.sigdict['cuda']) else 'CPU'
        if self.sigdict['skin_extractor'] == 'convexhull':
            sig_processing.set_skin_extractor(
                SkinExtractionConvexHull(target_device))
        elif self.sigdict['skin_extractor'] == 'faceparsing':
            sig_processing.set_skin_extractor(
                SkinExtractionFaceParsing(target_device))
        # set patches
        if self.sigdict['approach'] == 'patches':
            ldmks_list = ast.literal_eval(
                self.sigdict['landmarks_list'])
            if len(ldmks_list) > 0:
                sig_processing.set_landmarks(ldmks_list)
            if self.sigdict['patches'] == 'squares':
                # set squares patches side dimension
                sig_processing.set_square_patches_side(
                    float(self.sigdict['squares_dim']))
            elif self.sigdict['patches'] == 'rects':
                # set rects patches sides dimensions
                rects_dims = ast.literal_eval(
                    self.sigdict['rects_dims'])
                if len(rects_dims) > 0:
                    sig_processing.set_rect_patches_sides(
                        np.array(rects_dims, dtype=np.float32))
        # set sig-processing and skin-processing params
        SignalProcessingParams.RGB_LOW_TH = np.int32(
            self.sigdict['sig_color_low_threshold'])
        SignalProcessingParams.RGB_HIGH_TH = np.int32(
            self.sigdict['sig_color_high_threshold'])
        SkinProcessingParams.RGB_LOW_TH = np.int32(
            self.sigdict['skin_color_low_threshold'])
        SkinProcessingParams.RGB_HIGH_TH = np.int32(
            self.sigdict['skin_color_high_threshold'])

        # load all the videos
        if self.videoIdx == []:
            self.videoIdx = [int(v)
                             for v in range(len(dataset.videoFilenames))]

        # -- loop on videos
        for v in self.videoIdx:

            # -- verbose prints
            if '1' in str(verb):
                print("\n## videoID: %d" % (v))

            # -- ground-truth signal
            try:
                fname = dataset.getSigFilename(v)
                sigGT = dataset.readSigfile(fname)
            except:
                continue
            winSizeGT = int(self.sigdict['winSize'])
            bpmGT, timesGT = sigGT.getBPM(winSizeGT)

            # -- video file name
            videoFileName = dataset.getVideoFilename(v)
            print(videoFileName)
            fps = get_fps(videoFileName)

            sig_processing.set_total_frames(
                int(self.sigdict['tot_sec'])*fps)

            sig = []
            if str(self.sigdict['approach']) == 'hol':
                # SIG extraction with holistic
                sig = sig_processing.extract_holistic(videoFileName)
            elif str(self.sigdict['approach']) == 'patches':
                # SIG extraction with patches
                sig = sig_processing.extract_patches(
                    videoFileName, str(self.sigdict['patches']), str(self.sigdict['type']))

            # -- sig windowing
            windowed_sig, timesES = sig_windowing(
                sig, int(self.sigdict['winSize']), 1, fps)

            # -- loop on methods
            bvps_collection = []
            methods_collection = []
            for m in self.methods:
                if '1' in str(verb):
                    print("## Computing method: %s" % (str(m)))
                # -- PRE FILTERING
                filtered_windowed_sig = windowed_sig

                # -- color threshold - applied only with patches
                if str(self.sigdict['approach']) == 'patches':
                    filtered_windowed_sig = apply_filter(
                        windowed_sig,
                        rgb_filter_th,
                        params={'RGB_LOW_TH':  np.int32(self.bvpdict['color_low_threshold']),
                                'RGB_HIGH_TH': np.int32(self.bvpdict['color_high_threshold'])})

                # -- custom filters
                prefilter_list = ast.literal_eval(
                    self.methodsdict[m]['pre_filtering'])
                if len(prefilter_list) > 0:
                    for f in prefilter_list:
                        if '1' in str(verb):
                            print("  pre-filter: %s" % f)
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
                        filtered_windowed_sig = apply_filter(
                            filtered_windowed_sig, method_to_call, fps=fps, params=ast.literal_eval(fdict['params']))

                # -- BVP extraction
                if self.methodsdict[m]['path'] != 'None':
                    # custom path
                    spec = util.spec_from_file_location(
                        self.methodsdict[m]['name'], self.methodsdict[m]['path'])
                    mod = util.module_from_spec(spec)
                    spec.loader.exec_module(mod)
                    method_to_call = getattr(mod, self.methodsdict[m]['name'])
                else:
                    # package path
                    module = import_module(
                        'pyVHR.BVP.methods')
                    method_to_call = getattr(
                        module, self.methodsdict[m]['name'])
                bvps = RGB_sig_to_BVP(filtered_windowed_sig, fps,
                                      device_type=self.methodsdict[m]['device_type'], method=method_to_call, params=ast.literal_eval(self.methodsdict[m]['params']))

                # POST FILTERING
                postfilter_list = ast.literal_eval(
                    self.methodsdict[m]['post_filtering'])
                if len(postfilter_list) > 0:
                    for f in postfilter_list:
                        if '1' in str(verb):
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

                bvps_collection.append(bvps)
                methods_collection.append(m)

            # -- BPM extraction
            # first each method, then multi-method
            multi_method_coll = [e for e in bvps_collection]
            methods_collection.append("Multi-Method")
            for i in range(len(bvps_collection)+1):
                if i == len(bvps_collection): # last element is the multi-method
                    bvp_element = concatenate_BVPs(multi_method_coll)
                    if bvp_element == 0:
                        print("[ERROR] Multi method Concatenation can't be executed because BVP signals have different shapes!")
                        continue
                else: # use single methods
                    bvp_element = bvps_collection[i]

                bpmES = None
                if str(self.bpmdict['type']) == 'welch':
                    if bool(self.sigdict['cuda']):
                        bpmES = BVP_to_BPM_cuda(bvp_element, fps, minHz=float(
                            self.bpmdict['minHz']), maxHz=float(self.bpmdict['maxHz']))
                    else:
                        bpmES = BVP_to_BPM(bvp_element, fps, minHz=float(
                            self.bpmdict['minHz']), maxHz=float(self.bpmdict['maxHz']))
                elif str(self.bpmdict['type']) == 'psd_clustering':
                    if bool(self.sigdict['cuda']):
                        bpmES = BVP_to_BPM_PSD_clustering_cuda(bvp_element, fps, minHz=float(
                            self.bpmdict['minHz']), maxHz=float(self.bpmdict['maxHz']))
                    else:
                        bpmES = BVP_to_BPM_PSD_clustering(bvp_element, fps, minHz=float(
                            self.bpmdict['minHz']), maxHz=float(self.bpmdict['maxHz']))
                if bpmES is None:
                    print("[ERROR] BPM extraction error; check cfg params!")
                    continue
                # median BPM from multiple estimators BPM
                # this doesn't affect holistic approach
                #median_bpmES = multi_est_BPM_median(bpmES)
                median_bpmES, mad_bpmES = multi_est_BPM_median(bpmES)

                # -- error metrics
                RMSE, MAE, MAX, PCC, CCC = getErrors(median_bpmES, bpmGT, timesES, timesGT)

                # -- save results
                method_name = methods_collection[i]
                res.newDataSerie()
                res.addData('dataset', str(self.datasetdict['dataset']))
                res.addData('method', str(method_name))
                res.addData('videoIdx', v)
                res.addData('RMSE', RMSE)
                res.addData('MAE', MAE)
                res.addData('MAX', MAX)
                res.addData('PCC', PCC)
                res.addData('bpmGT', bpmGT)
                res.addData('bpmES', median_bpmES)
                res.addData('timeGT', timesGT)
                res.addData('timeES', timesES)
                res.addData('videoFilename', videoFileName)
                res.addDataSerie()

                if '1' in str(verb):
                    print("## Results for method: %s" % (str(method_name)))
                    printErrors(RMSE, MAE, MAX, PCC)

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
        self.sigdict = dict(self.parser['SIG'].items())
        self.bvpdict = dict(self.parser['BVP'].items())
        self.bpmdict = dict(self.parser['BPM'].items())

        # video idx list extraction
        if isinstance(ast.literal_eval(self.datasetdict['videoIdx']), list):
            self.videoIdx = [int(v) for v in ast.literal_eval(
                self.datasetdict['videoIdx'])]

        # load parameters for each methods
        self.methodsdict = {}
        self.methods = ast.literal_eval(self.bvpdict['methods'])
        for x in self.methods:
            self.methodsdict[x] = dict(self.parser[x].items())

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
    This class is used by :py:class:`pyVHR.analysis.multi_method_suite.MultiMethodSuite` to manage the results
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
        D['MAX'] = ''
        D['bpmGT'] = ''          # GT bpm
        D['bpmES'] = ''
        D['timeGT'] = ''            # GT bpm
        D['timeES'] = ''
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
