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
from pyVHR.utils.errors import getErrors, printErrors, displayErrors, get_SNR
from pyVHR.extraction.sig_processing import *
from pyVHR.extraction.sig_extraction_methods import *
from pyVHR.extraction.skin_extraction_methods import *
from pyVHR.BVP.BVP import *
from pyVHR.BPM.BPM import *
from pyVHR.BVP.methods import *
from pyVHR.BVP.filters import *
import time
from inspect import getmembers, isfunction
import os.path

class Pipeline():
    """ 
    This class runs the pyVHR pipeline on a single video or dataset
    """

    def __init__(self):
        pass

    def run_on_video(self, videoFileName, cuda=True, roi_method='convexhull', roi_approach='hol', method='cupy_POS', bpm_type='welch', pre_filt=False, post_filt=True, verb=True):
        """ 
        Runs the pipeline on a specific video file.

        Args:
            videoFileName:
                - The path to the video file to analyse
            cuda:
                - True - Enable computations on GPU
                - False - Use CPU only
            roi_method:
                - 'convexhull' - Uses MediaPipe's lanmarks to compute the convex hull of the face and segment the skin
                - 'faceparsing' - Uses BiseNet to parse face components and segment the skin
            roi_approach:
                - 'hol' - Use the Holistic approach (one single ROI defined as the whole face skin region of the subject)
                - 'patches' - Use multiple patches as Regions of Interest
            method:
                - One of the rPPG methods defined in pyVHR
            bpm_type:
                - the method for computing the BPM estimate on a time window
            pre_filt:
                - True - Use Band pass filtering on the windowed RGB signal
                - False - No pre-filtering
            post_filt:
                - True - Use Band pass filtering on the estimated BVP signal
                - False - No post-filtering
            verb:
               - False - not verbose
               - True - show the main steps  
        """
        ldmks_list = [2, 3, 4, 5, 6, 8, 9, 10, 18, 21, 32, 35, 36, 43, 46, 47, 48, 50, 54, 58, 67, 68, 69, 71, 92, 93, 101, 103, 104, 108, 109, 116, 117, 118, 123, 132, 134, 135, 138, 139, 142, 148, 149, 150, 151, 152, 182, 187, 188, 193, 197, 201, 205, 206, 207, 210, 211, 212, 216, 234, 248, 251, 262, 265, 266, 273, 277, 278, 280, 284, 288, 297, 299, 322, 323, 330, 332, 333, 337, 338, 345, 346, 361, 363, 364, 367, 368, 371, 377, 379, 411, 412, 417, 421, 425, 426, 427, 430, 432, 436]
        assert os.path.isfile(videoFileName), "\nThe provided video file does not exists!"
        
        sig_processing = SignalProcessing()
        av_meths = getmembers(pyVHR.BVP.methods, isfunction)
        available_methods = [am[0] for am in av_meths]

        assert method in available_methods, "\nrPPG method not recognized!!"

        if cuda:
            sig_processing.display_cuda_device()
            sig_processing.choose_cuda_device(0)
        
        # set skin extractor
        target_device = 'GPU' if cuda else 'CPU'
        if roi_method == 'convexhull':
            sig_processing.set_skin_extractor(
                SkinExtractionConvexHull(target_device))
        elif roi_method == 'faceparsing':
            sig_processing.set_skin_extractor(
                SkinExtractionFaceParsing(target_device))
        else:
            raise ValueError("Unknown 'roi_method'")
        
        assert roi_approach == 'patches' or roi_approach=='hol', "\nROI extraction approach not recognized!"
        
        # set patches
        if roi_approach == 'patches':
            #ldmks_list = ast.literal_eval(landmarks_list)
            #if len(ldmks_list) > 0:
            sig_processing.set_landmarks(ldmks_list)
            # set squares patches side dimension
            sig_processing.set_square_patches_side(28.0)
        
        # set sig-processing and skin-processing params
        SignalProcessingParams.RGB_LOW_TH = 75
        SignalProcessingParams.RGB_HIGH_TH = 230
        SkinProcessingParams.RGB_LOW_TH = 75
        SkinProcessingParams.RGB_HIGH_TH = 230

        if verb:
            print('\nProcessing Video: ' + videoFileName)
        fps = get_fps(videoFileName)
        sig_processing.set_total_frames(0)

        # -- ROI selection
        sig = []
        if roi_approach == 'hol':
            # SIG extraction with holistic
            sig = sig_processing.extract_holistic(videoFileName)
        elif roi_approach == 'patches':
            # SIG extraction with patches
            sig = sig_processing.extract_patches(videoFileName, 'squares', 'mean')

        # -- sig windowing
        windowed_sig, timesES = sig_windowing(sig, 6, 1, fps)

        # -- PRE FILTERING
        filtered_windowed_sig = windowed_sig

        # -- color threshold - applied only with patches
        if roi_approach == 'patches':
            filtered_windowed_sig = apply_filter(windowed_sig,
                                                 rgb_filter_th,
                                                 params={'RGB_LOW_TH':  75,
                                                         'RGB_HIGH_TH': 230})

        if pre_filt:
            module = import_module('pyVHR.BVP.filters')
            method_to_call = getattr(module, 'BPfilter')
            bvps = apply_filter(bvps, 
                                method_to_call, 
                                fps=fps, 
                                params={'minHz':0.65, 'maxHz':4.0, 'fps':'adaptive', 'order':6})
        if verb:
            print("\nBVP extraction with method: %s" % (method))

        # -- BVP Extraction
        module = import_module('pyVHR.BVP.methods')
        method_to_call = getattr(module, method)
        if 'cpu' in method:
            method_device = 'cpu'
        elif 'torch' in method:
            method_device = 'torch'
        elif 'cupy' in method:
            method_device = 'cuda'

        if 'POS' in method:
            pars = {'fps':'adaptive'}
        elif 'PCA' in method or 'ICA' in method:
            pars = {'component': 'all_comp'}
        else:
            pars = {}

        bvps = RGB_sig_to_BVP(windowed_sig, fps,
                              device_type=method_device, method=method_to_call, params=pars)
        
        # -- POST FILTERING
        if post_filt:
            module = import_module('pyVHR.BVP.filters')
            method_to_call = getattr(module, 'BPfilter')
            bvps = apply_filter(bvps, 
                                method_to_call, 
                                fps=fps, 
                                params={'minHz':0.65, 'maxHz':4.0, 'fps':'adaptive', 'order':6})

        if verb:
            print("\nBPM estimation with: %s" % (bpm_type))
        # -- BPM Estimation
        if bpm_type == 'welch':
            if cuda:
                bpmES = BVP_to_BPM_cuda(bvps, fps, minHz=0.65, maxHz=4.0)
            else:
                bpmES = BVP_to_BPM(bvps, fps, minHz=0.65, maxHz=4.0)
        elif bpm_type == 'psd_clustering':
            if cuda:
                bpmES = BVP_to_BPM_PSD_clustering_cuda(bvps, fps, minHz=0.65, maxHz=4.0)
            else:
                bpmES = BVP_to_BPM_PSD_clustering(bvps, fps, minHz=0.65, maxHz=4.0)
        else:
            raise ValueError("Unknown 'bpm_type'")

        # median BPM from multiple estimators BPM
        median_bpmES, mad_bpmES = multi_est_BPM_median(bpmES)

        if verb:
            print('\n...done!\n')

        return timesES, median_bpmES, mad_bpmES


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

        # -- SIG processing
        sig_processing = SignalProcessing()
        if eval(self.sigdict['cuda']):
            sig_processing.display_cuda_device()
            sig_processing.choose_cuda_device(int(self.sigdict['cuda_device']))
        # set skin extractor
        target_device = 'GPU' if eval(self.sigdict['cuda']) else 'CPU'
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
            winSizeGT = int(self.sigdict['winSize'])
            bpmGT, timesGT = sigGT.getBPM(winSizeGT)

            # -- video file name
            videoFileName = dataset.getVideoFilename(v)
            print(videoFileName)
            fps = get_fps(videoFileName)

            #Start chronometer
            #start_time = time.time()

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
            for m in self.methods:
                if verb:
                    print("## method: %s" % (str(m)))

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
                        if verb:
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
                    if eval(self.sigdict['cuda']):
                        bpmES = BVP_to_BPM_cuda(bvps, fps, minHz=float(
                            self.bpmdict['minHz']), maxHz=float(self.bpmdict['maxHz']))
                    else:
                        bpmES = BVP_to_BPM(bvps, fps, minHz=float(
                            self.bpmdict['minHz']), maxHz=float(self.bpmdict['maxHz']))
                elif self.bpmdict['type'] == 'psd_clustering':
                    if eval(self.sigdict['cuda']):
                        bpmES = BVP_to_BPM_PSD_clustering_cuda(bvps, fps, minHz=float(
                            self.bpmdict['minHz']), maxHz=float(self.bpmdict['maxHz']))
                    else:
                        bpmES = BVP_to_BPM_PSD_clustering(bvps, fps, minHz=float(
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
