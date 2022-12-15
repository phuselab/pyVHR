import configparser
import ast
from numpy.lib.arraysetops import isin
import pandas as pd
import numpy as np
from importlib import import_module, util
from pyVHR.datasets.dataset import datasetFactory
from pyVHR.utils.errors import getErrors, printErrors, displayErrors, BVP_windowing
from pyVHR.extraction.sig_processing import *
from pyVHR.extraction.sig_extraction_methods import *
from pyVHR.extraction.skin_extraction_methods import *
from pyVHR.BVP.BVP import *
from pyVHR.BPM.BPM import *
from pyVHR.BVP.methods import *
from pyVHR.BVP.filters import *
from inspect import getmembers, isfunction
import os.path
from pyVHR.deepRPPG.mtts_can import *
from pyVHR.deepRPPG.hr_cnn import *
from pyVHR.extraction.utils import *

class Pipeline():
    """ 
    This class runs the pyVHR pipeline on a single video or dataset
    """

    minHz = 0.65 # min heart frequency in Hz
    maxHz = 4.0  # max heart frequency in Hz

    def __init__(self):
        pass


    def run_on_video_multimethods(self, videoFileName, 
                    winsize, 
                    ldmks_list=None,
                    cuda=True, 
                    roi_method='convexhull', 
                    roi_approach='holistic', 
                    methods=['cpu_CHROM, cpu_POS, cpu_LGI'], 
                    estimate='holistic', 
                    movement_thrs=[10, 5, 2],
                    patch_size=30, 
                    RGB_LOW_HIGH_TH = (75,230),
                    Skin_LOW_HIGH_TH = (75, 230),
                    pre_filt=False, 
                    post_filt=True, 
                    verb=True):
        """ 
        Runs the pipeline on a specific video file.

        Args:
            videoFileName:
                - The video filenane to analyse
            winsize:
                - The size of the window in frame
            ldmks_list:
                - (default None) a list of MediaPipe's landmarks to use, the range is: [0:467]
            cuda:
                - True - Enable computations on GPU
            roi_method:
                - 'convexhull', uses MediaPipe's lanmarks to compute the convex hull on the face skin
                - 'faceparsing', uses BiseNet to parse face components and segment the skin
            roi_approach:
                - 'holistic', uses the holistic approach, i.e. the whole face skin
                - 'patches', uses multiple patches as Regions of Interest
            methods:
                - A collection of rPPG methods defined in pyVHR
            estimate:
                - if patches: 'medians', 'clustering', the method for BPM estimate on each window 
            movement_thrs:
                - Thresholds for movements filtering (eg.:[10, 5, 2])
            patch_size:
                - the size of the square patch, in pixels
            RGB_LOW_HIGH_TH: 
                - default (75,230), thresholds for RGB channels 
            Skin_LOW_HIGH_TH:
                - default (75,230), thresholds for skin pixel values 
            pre_filt:
                - True, uses bandpass filter on the windowed RGB signal
            post_filt:
                - True, uses bandpass filter on the estimated BVP signal
            verb:
                - True, shows the main steps  
        """

        # set landmark list
        if not ldmks_list:
            ldmks_list = [2, 3, 4, 5, 6, 8, 9, 10, 18, 21, 32, 35, 36, 43, 46, 47, 48, 50, 54, 58, 67, 68, 69, 71, 92, 93, 101, 103, 104, 108, 109, 116, 117, 118, 123, 132, 134, 135, 138, 139, 142, 148, 149, 150, 151, 152, 182, 187, 188, 193, 197, 201, 205, 206, 207, 210, 211, 212, 216, 234, 248, 251, 262, 265, 266, 273, 277, 278, 280, 284, 288, 297, 299, 322, 323, 330, 332, 333, 337, 338, 345, 346, 361, 363, 364, 367, 368, 371, 377, 379, 411, 412, 417, 421, 425, 426, 427, 430, 432, 436]
        
        # test video filename
        assert os.path.isfile(videoFileName), "The video file does not exists!"
        
        sig_processing = SignalProcessing()
        av_meths = getmembers(pyVHR.BVP.methods, isfunction)
        available_methods = [am[0] for am in av_meths]

        for m in methods:
            assert m in available_methods, "\nrPPG method not recognized!!"

        if cuda:
            sig_processing.display_cuda_device()
            sig_processing.choose_cuda_device(0)
        
        ## 1. set skin extractor
        target_device = 'GPU' if cuda else 'CPU'
        if roi_method == 'convexhull':
            sig_processing.set_skin_extractor(SkinExtractionConvexHull(target_device))
        elif roi_method == 'faceparsing':
            sig_processing.set_skin_extractor(SkinExtractionFaceParsing(target_device))
        else:
            raise ValueError("Unknown 'roi_method'")
              
        ## 2. set patches
        if roi_approach == 'patches':
            sig_processing.set_landmarks(ldmks_list)
            sig_processing.set_square_patches_side(np.float(patch_size))
        
        # set sig-processing and skin-processing params
        SignalProcessingParams.RGB_LOW_TH = RGB_LOW_HIGH_TH[0]
        SignalProcessingParams.RGB_HIGH_TH = RGB_LOW_HIGH_TH[1]
        SkinProcessingParams.RGB_LOW_TH = Skin_LOW_HIGH_TH[0]
        SkinProcessingParams.RGB_HIGH_TH = Skin_LOW_HIGH_TH[1]

        if verb:
            print('\nProcessing Video ' + videoFileName)
        fps = get_fps(videoFileName)
        sig_processing.set_total_frames(0)

        ## 3. ROI selection
        if verb:
            print('\nRoi processing...')
        sig = []
        if roi_approach == 'holistic':
            # SIG extraction with holistic
            sig = sig_processing.extract_holistic(videoFileName)
        elif roi_approach == 'patches':
            # SIG extraction with patches
            sig = sig_processing.extract_patches(videoFileName, 'squares', 'mean')
        if verb:
            print(' - Extraction approach: ' + roi_approach)

        ## 4. sig windowing
        windowed_sig, timesES = sig_windowing(sig, winsize, 1, fps)
        if verb:
            print(f' - Number of windows: {len(windowed_sig)}')
            print(' - Win size: (#ROI, #landmarks, #frames) = ', windowed_sig[0].shape)


        ## 5. PRE FILTERING
        if verb:
            print('\nPre filtering...')
        filtered_windowed_sig = windowed_sig
        
        # -- color threshold - applied only with patches
        #if roi_approach == 'patches':
        #    filtered_windowed_sig = apply_filter(windowed_sig,
        #                                        rgb_filter_th,
        #                                        params={'RGB_LOW_TH': RGB_LOW_HIGH_TH[0],
        #                                                'RGB_HIGH_TH': RGB_LOW_HIGH_TH[1]})

        if pre_filt:
            module = import_module('pyVHR.BVP.filters')
            method_to_call = getattr(module, 'BPfilter')
            filtered_windowed_sig = apply_filter(filtered_windowed_sig,
                                                    method_to_call, 
                                                    fps=fps, 
                                                    params={'minHz':Pipeline.minHz, 
                                                            'maxHz':Pipeline.maxHz, 
                                                            'fps':'adaptive', 
                                                            'order':6})
        if verb:
            print(f' - Pre-filter applied: {method_to_call.__name__}')

        ## 6. BVP extraction multimethods
        bvps_win = []
        for method in methods:
            if verb:
                print("\nBVP extraction...")
                print(" - Extraction method: " + method)
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

            bvps_win_m = RGB_sig_to_BVP(filtered_windowed_sig, 
                                    fps, device_type=method_device, 
                                    method=method_to_call, params=pars)

            ## 7. POST FILTERING
            if post_filt:
                module = import_module('pyVHR.BVP.filters')
                method_to_call = getattr(module, 'BPfilter')
                bvps_win_m = apply_filter(bvps_win_m, 
                                    method_to_call, 
                                    fps=fps, 
                                    params={'minHz':Pipeline.minHz, 'maxHz':Pipeline.maxHz, 'fps':'adaptive', 'order':6})
            if verb:
                print(f' - Post-filter applied: {method_to_call.__name__}')
            # collect
            if len(bvps_win) == 0:
                bvps_win = bvps_win_m
            else:
                for i in range(len(bvps_win_m)):
                    bvps_win[i] = np.concatenate((bvps_win[i], bvps_win_m[i]))
                    if i == 0: print(bvps_win[i].shape)
            

        ## 8. BPM extraction
        if verb:
            print("\nBPM estimation...")  
            print(f" - roi appproach: {roi_approach}")  

        if roi_approach == 'holistic':
            if cuda:
                bpmES = BVP_to_BPM_cuda(bvps_win, fps, minHz=Pipeline.minHz, maxHz=Pipeline.maxHz)
            else:
                bpmES = BVP_to_BPM(bvps_win, fps, minHz=Pipeline.minHz, maxHz=Pipeline.maxHz)

        elif roi_approach == 'patches':
            if estimate == 'clustering':
                #if cuda and False:
                #    bpmES = BVP_to_BPM_PSD_clustering_cuda(bvps_win, fps, minHz=Pipeline.minHz, maxHz=Pipeline.maxHz)
                #else:
                #bpmES = BPM_clustering(sig_processing, bvps_win, winsize, movement_thrs=[15, 15, 15], fps=fps, opt_factor=0.5)
                ma = MotionAnalysis(sig_processing, winsize, fps)
                bpmES = BPM_clustering(ma, bvps_win, fps, winsize, movement_thrs=movement_thrs, opt_factor=0.5)



            elif estimate == 'median':
                if cuda:
                    bpmES = BVP_to_BPM_cuda(bvps_win, fps, minHz=Pipeline.minHz, maxHz=Pipeline.maxHz)
                else:
                    bpmES = BVP_to_BPM(bvps_win, fps, minHz=Pipeline.minHz, maxHz=Pipeline.maxHz)
                bpmES,_ = BPM_median(bpmES)
            if verb:
                print(f" - BPM estimation with: {estimate}")
        else:
            raise ValueError("Estimation approach unknown!")

        if verb:
            print('\n...done!\n')

        return bvps_win, timesES, bpmES

    def run_on_video(self, videoFileName, 
                    winsize, 
                    ldmks_list=None,
                    cuda=True, 
                    roi_method='convexhull', 
                    roi_approach='holistic', 
                    method='cupy_POS', 
                    estimate='holistic', 
                    movement_thrs=[10, 5, 2],
                    patch_size=30, 
                    RGB_LOW_HIGH_TH = (75,230),
                    Skin_LOW_HIGH_TH = (75, 230),
                    pre_filt=False, 
                    post_filt=True, 
                    verb=True):
        """ 
        Runs the pipeline on a specific video file.

        Args:
            videoFileName:
                - The video filenane to analyse
            winsize:
                - The size of the window in frame
            ldmks_list:
                - (default None) a list of MediaPipe's landmarks to use, the range is: [0:467]
            cuda:
                - True - Enable computations on GPU
            roi_method:
                - 'convexhull', uses MediaPipe's lanmarks to compute the convex hull on the face skin
                - 'faceparsing', uses BiseNet to parse face components and segment the skin
            roi_approach:
                - 'holistic', uses the holistic approach, i.e. the whole face skin
                - 'patches', uses multiple patches as Regions of Interest
            method:
                - One of the rPPG methods defined in pyVHR
            estimate:
                - if patches: 'medians', 'clustering', the method for BPM estimate on each window 
            movement_thrs:
                - Thresholds for movements filtering (eg.:[10, 5, 2])
            patch_size:
                - the size of the square patch, in pixels
            RGB_LOW_HIGH_TH: 
                - default (75,230), thresholds for RGB channels 
            Skin_LOW_HIGH_TH:
                - default (75,230), thresholds for skin pixel values 
            pre_filt:
                - True, uses bandpass filter on the windowed RGB signal
            post_filt:
                - True, uses bandpass filter on the estimated BVP signal
            verb:
                - True, shows the main steps  
        """

        # set landmark list
        if not ldmks_list:
            ldmks_list = [2, 3, 4, 5, 6, 8, 9, 10, 18, 21, 32, 35, 36, 43, 46, 47, 48, 50, 54, 58, 67, 68, 69, 71, 92, 93, 101, 103, 104, 108, 109, 116, 117, 118, 123, 132, 134, 135, 138, 139, 142, 148, 149, 150, 151, 152, 182, 187, 188, 193, 197, 201, 205, 206, 207, 210, 211, 212, 216, 234, 248, 251, 262, 265, 266, 273, 277, 278, 280, 284, 288, 297, 299, 322, 323, 330, 332, 333, 337, 338, 345, 346, 361, 363, 364, 367, 368, 371, 377, 379, 411, 412, 417, 421, 425, 426, 427, 430, 432, 436]
        
        # test video filename
        assert os.path.isfile(videoFileName), "The video file does not exists!"
        
        sig_processing = SignalProcessing()
        av_meths = getmembers(pyVHR.BVP.methods, isfunction)
        available_methods = [am[0] for am in av_meths]

        assert method in available_methods, "\nrPPG method not recognized!!"

        if cuda:
            sig_processing.display_cuda_device()
            sig_processing.choose_cuda_device(0)
        
        ## 1. set skin extractor
        target_device = 'GPU' if cuda else 'CPU'
        if roi_method == 'convexhull':
            sig_processing.set_skin_extractor(SkinExtractionConvexHull(target_device))
        elif roi_method == 'faceparsing':
            sig_processing.set_skin_extractor(SkinExtractionFaceParsing(target_device))
        else:
            raise ValueError("Unknown 'roi_method'")
              
        ## 2. set patches
        if roi_approach == 'patches':
            sig_processing.set_landmarks(ldmks_list)
            sig_processing.set_square_patches_side(np.float(patch_size))
        
        # set sig-processing and skin-processing params
        SignalProcessingParams.RGB_LOW_TH = RGB_LOW_HIGH_TH[0]
        SignalProcessingParams.RGB_HIGH_TH = RGB_LOW_HIGH_TH[1]
        SkinProcessingParams.RGB_LOW_TH = Skin_LOW_HIGH_TH[0]
        SkinProcessingParams.RGB_HIGH_TH = Skin_LOW_HIGH_TH[1]

        if verb:
            print('\nProcessing Video ' + videoFileName)
        fps = get_fps(videoFileName)
        sig_processing.set_total_frames(0)

        ## 3. ROI selection
        if verb:
            print('\nRoi processing...')
        sig = []
        if roi_approach == 'holistic':
            # SIG extraction with holistic
            sig = sig_processing.extract_holistic(videoFileName)
        elif roi_approach == 'patches':
            # SIG extraction with patches
            sig = sig_processing.extract_patches(videoFileName, 'squares', 'mean')
        if verb:
            print(' - Extraction approach: ' + roi_approach)

        ## 4. sig windowing
        windowed_sig, timesES = sig_windowing(sig, winsize, 1, fps)
        if verb:
            print(f' - Number of windows: {len(windowed_sig)}')
            print(' - Win size: (#ROI, #landmarks, #frames) = ', windowed_sig[0].shape)


        ## 5. PRE FILTERING
        if verb:
            print('\nPre filtering...')
        filtered_windowed_sig = windowed_sig
        
        # -- color threshold - applied only with patches
        #if roi_approach == 'patches':
        #    filtered_windowed_sig = apply_filter(windowed_sig,
        #                                        rgb_filter_th,
        #                                        params={'RGB_LOW_TH': RGB_LOW_HIGH_TH[0],
        #                                                'RGB_HIGH_TH': RGB_LOW_HIGH_TH[1]})

        if pre_filt:
            module = import_module('pyVHR.BVP.filters')
            method_to_call = getattr(module, 'BPfilter')
            filtered_windowed_sig = apply_filter(filtered_windowed_sig,
                                                    method_to_call, 
                                                    fps=fps, 
                                                    params={'minHz':Pipeline.minHz, 
                                                            'maxHz':Pipeline.maxHz, 
                                                            'fps':'adaptive', 
                                                            'order':6})
        if verb:
            print(f' - Pre-filter applied: {method_to_call.__name__}')

        ## 6. BVP extraction
        if verb:
            print("\nBVP extraction...")
            print(" - Extraction method: " + method)
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

        bvps_win = RGB_sig_to_BVP(filtered_windowed_sig, 
                                fps, device_type=method_device, 
                                method=method_to_call, params=pars)

        ## 7. POST FILTERING
        if post_filt:
            module = import_module('pyVHR.BVP.filters')
            method_to_call = getattr(module, 'BPfilter')
            bvps_win = apply_filter(bvps_win, 
                                method_to_call, 
                                fps=fps, 
                                params={'minHz':Pipeline.minHz, 'maxHz':Pipeline.maxHz, 'fps':'adaptive', 'order':6})
        if verb:
            print(f' - Post-filter applied: {method_to_call.__name__}')

        ## 8. BPM extraction
        if verb:
            print("\nBPM estimation...")  
            print(f" - roi appproach: {roi_approach}")  

        if roi_approach == 'holistic':
            if cuda:
                bpmES = BVP_to_BPM_cuda(bvps_win, fps, minHz=Pipeline.minHz, maxHz=Pipeline.maxHz)
            else:
                bpmES = BVP_to_BPM(bvps_win, fps, minHz=Pipeline.minHz, maxHz=Pipeline.maxHz)

        elif roi_approach == 'patches':
            if estimate == 'clustering':
                #if cuda and False:
                #    bpmES = BVP_to_BPM_PSD_clustering_cuda(bvps_win, fps, minHz=Pipeline.minHz, maxHz=Pipeline.maxHz)
                #else:
                #bpmES = BPM_clustering(sig_processing, bvps_win, winsize, movement_thrs=[15, 15, 15], fps=fps, opt_factor=0.5)
                ma = MotionAnalysis(sig_processing, winsize, fps)
                bpmES = BPM_clustering(ma, bvps_win, fps, winsize, movement_thrs=movement_thrs, opt_factor=0.5)



            elif estimate == 'median':
                if cuda:
                    bpmES = BVP_to_BPM_cuda(bvps_win, fps, minHz=Pipeline.minHz, maxHz=Pipeline.maxHz)
                else:
                    bpmES = BVP_to_BPM(bvps_win, fps, minHz=Pipeline.minHz, maxHz=Pipeline.maxHz)
                bpmES,_ = BPM_median(bpmES)
            if verb:
                print(f" - BPM estimation with: {estimate}")
        else:
            raise ValueError("Estimation approach unknown!")

        if verb:
            print('\n...done!\n')

        return bvps_win, timesES, bpmES

    def run_on_dataset(self, configFilename, verb=True):
        """ 
        Like the 'run_on_video' function, it runs on all videos of a specific 
        dataset as specified by the loaded configuration file.

        Args:
            configFilename:
                - The path to the configuration file
            verb:
                - False - not verbose
                - True - show the main steps
        """
        # -- cfg file  
        self.configFilename = configFilename
        self.parse_cfg(self.configFilename)
        
        # -- cfg parser
        parser = configparser.ConfigParser(inline_comment_prefixes=('#', ';'))
        parser.optionxform = str
        if not parser.read(self.configFilename):
            raise FileNotFoundError(self.configFilename)
        if verb:
            self.__verbose('a')

        # -- dataset & cfg params
        if 'path' in self.datasetdict and self.datasetdict['path'] != 'None':
            dataset = datasetFactory(self.datasetdict['dataset'], 
                                    videodataDIR=self.datasetdict['videodataDIR'], 
                                    BVPdataDIR=self.datasetdict['BVPdataDIR'], 
                                    path=self.datasetdict['path'])
        else:
            dataset = datasetFactory(self.datasetdict['dataset'], 
                                    videodataDIR=self.datasetdict['videodataDIR'], 
                                    BVPdataDIR=self.datasetdict['BVPdataDIR'])

        # -- catch data (object)
        res = TestResult()

        # -- SIG processing
        sig_processing = SignalProcessing()
        if eval(self.sigdict['cuda']):
            sig_processing.display_cuda_device()
            sig_processing.choose_cuda_device(int(self.sigdict['cuda_device']))
        if verb:
            print(f" -  cuda device: {self.sigdict['cuda']}")

        ## 1. set skin extractor
        target_device = 'GPU' if eval(self.sigdict['cuda']) else 'CPU'
        if self.sigdict['skin_extractor'] == 'convexhull':
            sig_processing.set_skin_extractor(
                SkinExtractionConvexHull(target_device))
        elif self.sigdict['skin_extractor'] == 'faceparsing':
            sig_processing.set_skin_extractor(
                SkinExtractionFaceParsing(target_device))
        else:
            raise ValueError("Unknown roi method extraction!")
        if verb:
            print(f" -  skin extractor: {self.sigdict['skin_extractor']}")
 
        ## 2. set patches
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
        if verb:
            print(f" -  ROI approach: {self.sigdict['approach']}")
        
        # set sig-processing and skin-processing params
        SignalProcessingParams.RGB_LOW_TH = np.int32(
            self.sigdict['sig_color_low_threshold'])
        SignalProcessingParams.RGB_HIGH_TH = np.int32(
            self.sigdict['sig_color_high_threshold'])
        SkinProcessingParams.RGB_LOW_TH = np.int32(
            self.sigdict['skin_color_low_threshold'])
        SkinProcessingParams.RGB_HIGH_TH = np.int32(
            self.sigdict['skin_color_high_threshold'])

        # set video idx
        self.videoIdx = [int(v) for v in range(len(dataset.videoFilenames))]

        # -- loop on videos
        for v in self.videoIdx:

            ####if v != 5: continue     ##### to remove

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

            sig_processing.set_total_frames(
                int(self.sigdict['tot_sec'])*fps)

            ## 3. ROI selection
            sig = []
            if str(self.sigdict['approach']) == 'holistic':
                # mean extraction with holistic
                sig = sig_processing.extract_holistic(videoFileName)
            elif str(self.sigdict['approach']) == 'patches':
                # mean extraction with patches
                sig = sig_processing.extract_patches(
                    videoFileName, str(self.sigdict['patches']), str(self.sigdict['type']))

            ## 4. sig windowing
            windowed_sig, timesES = sig_windowing(sig, int(self.sigdict['winSize']), 1, fps)

            # -- loop on methods
            for m in self.methods:
                if verb:
                    app = str(self.sigdict['approach'])
                    print(f'## method: {str(m)} ({app})')

                ## 5. PRE FILTERING
                filtered_windowed_sig = windowed_sig

                # -- color threshold - applied only with patches
                #if str(self.sigdict['approach']) == 'patches':
                #    filtered_windowed_sig = apply_filter(windowed_sig, rgb_filter_th,
                #        params={'RGB_LOW_TH':  np.int32(self.bvpdict['color_low_threshold']),
                #                'RGB_HIGH_TH': np.int32(self.bvpdict['color_high_threshold'])})

                # -- custom filters
                prefilter_list = ast.literal_eval(self.methodsdict[m]['pre_filtering'])
                if len(prefilter_list) > 0:
                    for f in prefilter_list:
                        if verb:
                            print("  pre-filter: %s" % f)
                        fdict = dict(parser[f].items())
                        if fdict['path'] != 'None':
                            # custom path
                            spec = util.spec_from_file_location(fdict['name'], fdict['path'])
                            mod = util.module_from_spec(spec)
                            spec.loader.exec_module(mod)
                            method_to_call = getattr(mod, fdict['name'])
                        else:
                            # package path
                            module = import_module('pyVHR.BVP.filters')
                            method_to_call = getattr(module, fdict['name'])
                        filtered_windowed_sig = apply_filter(filtered_windowed_sig, method_to_call, fps=fps, params=ast.literal_eval(fdict['params']))

                ## 6. BVP extraction
                if self.methodsdict[m]['path'] != 'None':
                    # custom path
                    spec = util.spec_from_file_location(self.methodsdict[m]['name'], self.methodsdict[m]['path'])
                    mod = util.module_from_spec(spec)
                    spec.loader.exec_module(mod)
                    method_to_call = getattr(mod, self.methodsdict[m]['name'])
                else:
                    # package path
                    module = import_module('pyVHR.BVP.methods')
                    method_to_call = getattr(module, self.methodsdict[m]['name'])
                bvps_win = RGB_sig_to_BVP(filtered_windowed_sig, fps,
                                      device_type=self.methodsdict[m]['device_type'], 
                                      method=method_to_call, 
                                      params=ast.literal_eval(self.methodsdict[m]['params']))

                ## 7. POST FILTERING
                postfilter_list = ast.literal_eval(self.methodsdict[m]['post_filtering'])
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
                            method_to_call = getattr(mod, fdict['name'])
                        else:
                            # package path
                            module = import_module('pyVHR.BVP.filters')
                            method_to_call = getattr(module, fdict['name'])
                        
                        bvps_win = apply_filter(bvps_win, method_to_call, fps=fps, params=ast.literal_eval(fdict['params']))

                ## 8. BPM extraction
                MAD = []
                if self.bpmdict['estimate'] == 'holistic' or self.bpmdict['estimate'] == 'median':
                    if eval(self.sigdict['cuda']):
                        bpmES = BVP_to_BPM_cuda(bvps_win, fps, minHz=float(
                            self.bpmdict['minHz']), maxHz=float(self.bpmdict['maxHz']))
                    else:
                        bpmES = BVP_to_BPM(bvps_win, fps, minHz=float(
                            self.bpmdict['minHz']), maxHz=float(self.bpmdict['maxHz']))
                      
                    if self.bpmdict['estimate'] == 'median':
                        # median BPM from multiple estimators BPM
                        bpmES, MAD = BPM_median(bpmES)

                elif self.bpmdict['estimate'] == 'clustering':
                    # if eval(self.sigdict['cuda']):
                    #     bpmES = BVP_to_BPM_PSD_clustering_cuda(bvps_win, fps, minHz=float(
                    #         self.bpmdict['minHz']), maxHz=float(self.bpmdict['maxHz']))
                    # else:
                    #bpmES = BPM_clustering(sig_processing, bvps_win, winSizeGT, movement_thrs=[15, 15, 15], fps=fps, opt_factor=0.5)
                    ma = MotionAnalysis(sig_processing, winSizeGT, fps)
                    mthrs = self.bpmdict['movement_thrs']
                    mthrs = mthrs.replace('[', '')
                    mthrs = mthrs.replace(']', '')
                    movement_thrs = [float(i) for i in mthrs.split(",")]
                    bpmES = BPM_clustering(ma, bvps_win, fps, winSizeGT, movement_thrs=movement_thrs, opt_factor=0.5)
              

                ## 9. error metrics
                RMSE, MAE, MAX, PCC, CCC, SNR = getErrors(bvps_win, fps, bpmES, bpmGT, timesES, timesGT)

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
                res.addData('MAD', MAD)
                res.addData('bpmGT', bpmGT)
                res.addData('bpmES', bpmES)
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

        self.parser = configparser.ConfigParser(inline_comment_prefixes=('#', ';'))
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

class DeepPipeline(Pipeline):
    """ 
    This class runs the pyVHR Deep pipeline on a single video or dataset
    """

    def __init__(self):
        pass

    def run_on_video(self, videoFileName, cuda=True, method='MTTS_CAN', bpm_type='welch', post_filt=False, verb=True, crop_face=False):
        """ 
        Runs the pipeline on a specific video file.

        Args:
            videoFileName:
                - The path to the video file to analyse
            cuda:
                - True - Enable computations on GPU
                - False - Use CPU only
            method:
                - One of the rPPG methods defined in pyVHR
            bpm_type:
                - the method for computing the BPM estimate on a time window
            post_filt:
                - True - Use Band pass filtering on the estimated BVP signal
                - False - No post-filtering
            verb:
               - False - not verbose
               - True - show the main steps  
        """

        if verb:
            print('\nProcessing Video: ' + videoFileName)
        fps = get_fps(videoFileName)
        wsize = 6
        
        sp = SignalProcessing()
        frames = sp.extract_raw(videoFileName)
        print('Frames shape:', frames.shape)

        # -- BVP extraction
        if verb:
            print("\nBVP extraction with method: %s" % (method))
        if method == 'MTTS_CAN':
            bvps_pred = MTTS_CAN_deep(frames, fps, verb=1, filter_pred=True)
            bvps, timesES = BVP_windowing(bvps_pred, wsize, fps, stride=1)
        elif method == 'HR_CNN':
            bvps_pred = HR_CNN_bvp_pred(frames)
            bvps, timesES = BVP_windowing(bvps_pred, wsize, fps, stride=1)
        else:
            print("Deep Method unsupported!")
            return

        if post_filt:
            module = import_module('pyVHR.BVP.filters')
            method_to_call = getattr(module, 'BPfilter')
            bvps = apply_filter(bvps, 
                                method_to_call, 
                                fps=fps, 
                                params={'minHz':0.65, 'maxHz':4.0, 'fps':'adaptive', 'order':6})

        # -- BPM extraction
        if verb:
            print("\nBPM estimation with: %s" % (bpm_type))
        if bpm_type == 'welch':
            if cuda:
                bpmES = BVP_to_BPM_cuda(bvps, fps, minHz=minHz, maxHz=maxHz)
            else:
                bpmES = BVP_to_BPM(bvps, fps, minHz=minHz, maxHz=maxHz)
        else:
            raise ValueError("The only 'bpm_type' supported for deep models is 'welch'")
           
        # median BPM from multiple estimators BPM
        median_bpmES, mad_bpmES = BPM_median(bpmES)

        if verb:
            print('\n...done!\n')

        return timesES, median_bpmES

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

            # -- loop on methods
            for m in self.methods:
                if verb:
                    print("## method: %s" % (str(m)))

                # -- BVP extraction
                if str(m) == 'MTTS_CAN':
                    bvps_pred = MTTS_CAN_deep(frames, fps, verb=1, filter_pred=True)
                    bvps, timesES = BVP_windowing(bvps_pred, winSizeGT, fps, stride=1)
                elif str(m) == 'HR_CNN':
                    bvps_pred = HR_CNN_bvp_pred(frames)
                    bvps, timesES = BVP_windowing(bvps_pred, wsize, fps, stride=1)
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
                        bvps_win = apply_filter(bvps_win, method_to_call, fps=fps, params=ast.literal_eval(fdict['params']))

                # -- BPM extraction
                if self.bpmdict['type'] == 'welch':
                    bpmES = BVP_to_BPM_cuda(bvps_win, fps, minHz=float(
                        self.bpmdict['minHz']), maxHz=float(self.bpmdict['maxHz']))
                elif self.bpmdict['type'] == 'clustering':
                    bpmES = BVP_to_BPM_PSD_clustering_cuda(bvps_win, fps, minHz=float(
                        self.bpmdict['minHz']), maxHz=float(self.bpmdict['maxHz']))
                   
                # median BPM from multiple estimators BPM
                median_bpmES, mad_bpmES = BPM_median(bpmES)

                # -- error metrics
                RMSE, MAE, MAX, PCC, CCC, SNR = getErrors(bvps_win, fps, median_bpmES, bpmGT, timesES, timesGT)

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
    This class manages the results on a given dataset using multiple rPPG methods.
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
            self.dataFrame = self.dataFrame.append(self.dict, ignore_index=True)

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
        D['MAD'] = ''
        D['bpmGT'] = ''             # GT bpm
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
        if outFilename is None:
            outFilename = "testResults.h5"
        else:
            self.outFilename = outFilename

        # -- save data
        self.dataFrame.to_hdf(outFilename, key='self.dataFrame', mode='w')
