import os
import dlib, skvideo.io
import numpy as np
import cv2
import re
import warnings
import matplotlib.pyplot as plt
from matplotlib import patches
from .pyramid import *
from ..utils import printutils
from ..utils.SkinDetect import SkinDetect

class Video:
    """
    Basic class for extracting ROIs from video frames
    """

    facePadding = 0.2      # dlib param for padding
    filenameCompressed = "croppedFaces.npz"   # filename to store on disk
    saveCropFaces = True   # enable the storage on disk of the cropped faces
    loadCropFaces = True   # enable the loading of cropped faces from disk


    def __init__(self, filename, verb=0):
        self.filename = filename
        self.faces = np.array([])       # empty array of cropped faces (RGB)
        self.processedFaces = np.array([])
        self.faceSignal = np.array([])   # empty array of face signals (RGB) after roi/skin extraction
        
        self.verb = verb
        self.cropSize = [150,150]       # param for cropping
        self.typeROI = 'rect'           # type of rois between ['rect', 'skin']
        self.detector = 'mtcnn'
        
        self.doEVM = False
        self.EVMalpha = 20
        self.EVMlevels = 3
        self.EVMlow = .8
        self.EVMhigh = 4

        self.rectCoords = [[0,0, self.cropSize[0], self.cropSize[1]]]  # default 'rect' roi coordinates
        self.skinThresh_fix = [40, 80]# default min values of Sauturation and Value (HSV) for 'skin' roi
        self.skinThresh_adapt = 0.2

    def getCroppedFaces(self, detector='mtcnn', extractor='skvideo'):
        """ Time is in seconds"""

        # -- check if cropped faces already exists on disk
        path, name = os.path.split(self.filename)
        filenamez = path + '/' + self.filenameCompressed
     
        self.detector = detector
        self.extractor = extractor

        # -- if compressed exists... load it
        if self.loadCropFaces and os.path.isfile(filenamez):
            self.cropped = True
            data = np.load(filenamez, allow_pickle=True)
            self.faces = data['a']
            self.numFrames = int(data['b'])
            self.frameRate = int(data['c'])
            self.height = int(data['d'])
            self.width = int(data['e'])
            self.duration = float(data['f'])
            self.codec = data['g']
            self.detector = data['h']
            self.extractor = data['i']
            self.cropSize = self.faces[0].shape

            if self.detector != detector:
                warnings.warn("\nWARNING!! Requested detector method is different from the saved one\n")

        # -- if compressed does not exist, load orig. video and extract faces
        else:
            self.cropped = False

            # if the video signal is stored in video container
            if os.path.isfile(self.filename):
                # -- metadata
                metadata = skvideo.io.ffprobe(self.filename)
                self.numFrames = int(eval(metadata["video"]["@nb_frames"]))
                self.height = int(eval(metadata["video"]["@height"]))
                self.width = int(eval(metadata["video"]["@width"]))
                self.frameRate = int(np.round(eval(metadata["video"]["@avg_frame_rate"])))
                self.duration = float(eval(metadata["video"]["@duration"]))
                self.codec = metadata["video"]["@codec_name"]
                # -- load video on a ndarray with skvideo or openCV
                video = None
                if extractor == 'opencv':
                    video = self.__opencvRead()
                else:
                    video = skvideo.io.vread(self.filename)
                 
            # else if the video signal is stored as single frames
            else:    #elif os.path.isdir(self.filename):
                # -- load frames on a ndarray
                self.path = path
                video = self.__loadFrames()
                self.numFrames = len(video)
                self.height = video[0].shape[0]
                self.width = video[0].shape[1]
                self.frameRate = 30  ###### <<<<----- TO SET MANUALLY ####
                self.duration = self.numFrames/self.frameRate
                self.codec = 'raw'

            # -- extract faces and resize
            print('\n\n' + detector + '\n\n')
            self.__extractFace(video, method=detector)

            # -- store cropped faces on disk
            if self.saveCropFaces:
                np.savez_compressed(filenamez, a=self.faces,
                                    b=self.numFrames, c=self.frameRate,
                                    d=self.height, e=self.width,
                                    f=self.duration, g=self.codec,
                                    h=self.detector, i=self.extractor)

        if '1' in str(self.verb):
            self.printVideoInfo()
            if not self.cropped: 
                print('      Extracted faces: not found! Detecting...')
            else:
                print('      Extracted faces: found! Loading...')

    def setMask(self, typeROI='rect', 
                rectCoords=None, rectRegions=None, 
                skinThresh_fix=None, skinThresh_adapt=None):
        self.typeROI = typeROI
        if self.typeROI == 'rect':
            if rectCoords is not None:
                # List of rectangular ROIs: [[x0,y0,w0,h0],...,[xk,yk,wk,hk]]
                self.rectCoords = rectCoords
            elif rectRegions is not None:
                # List of rectangular regions: ['forehead', 'lcheek', 'rcheek', 'nose']
                self.rectCoords = self.__rectRegions2Coord(rectRegions)
        elif self.typeROI == 'skin_adapt' and skinThresh_adapt is not None:
            # Skin limits for HSV
            self.skinThresh_adapt = skinThresh_adapt
        elif self.typeROI == 'skin_fix' and skinThresh_fix is not None:
            # Skin limits for HSV
            self.skinThresh_fix = skinThresh_fix
        else:
            raise ValueError('Unrecognized type of ROI provided.')

    def extractSignal(self, frameSubset, count=None):
        if self.typeROI == 'rect':
            return self.__extractRectSignal(frameSubset)

        elif self.typeROI == 'skin_adapt' or self.typeROI == 'skin_fix':
            return self.__extractSkinSignal(frameSubset, count)

    def setEVM(self, enable=True, alpha=20, levels=3, low=.8, high=4):
        """Eulerian Video Magnification"""

        #rawFaces = self.faces
        #gaussFaces = gaussian_video(rawFaces, levels=levels)
        #filtered = temporal_ideal_filter(gaussFaces, low, high, self.frameRate)
        #amplified = alpha * filtered
        #self.faces = reconstruct_video_g(amplified, rawFaces, levels=levels)
        self.doEVM = enable

        if enable is True:
            self.EVMalpha = alpha
            self.EVMlevels = levels
            self.EVMlow = low
            self.EVMhigh = high

    def applyEVM(self):
        vid_data = gaussian_video(self.faces, self.EVMlevels)
        vid_data = temporal_bandpass_filter(vid_data, self.frameRate, 
                                            freq_min=self.EVMlow, 
                                            freq_max=self.EVMhigh)
        vid_data *= self.EVMalpha
        self.processedFaces = combine_pyramid_and_save(vid_data, 
                                                       self.faces, 
                                                       enlarge_multiple=3, 
                                                       fps=self.frameRate)
        
    def getMeanRGB(self):

        n_frames = len(self.faceSignal)
        n_roi = len(self.faceSignal[0])
        rgb = np.zeros([3, n_frames])

        for i in range(n_frames):
            mean_rgb = 0

            for roi in self.faceSignal[i]:
                idx = roi!=0
                idx2 = np.logical_and(np.logical_and(idx[:,:,0], idx[:,:,1]), idx[:,:,2])
                roi = roi[idx2]
                if len(roi)==0:
                    mean_rgb += 0
                else:
                    mean_rgb += np.mean(roi, axis=0)
                
            rgb[:,i] = mean_rgb/n_roi
        return rgb
    
    def printVideoInfo(self):
        print('\n   * Video filename: %s' %self.filename)
        print('         Total frames: %s' %self.numFrames)
        print('             Duration: %s (sec)' %np.round(self.duration,2))
        print('           Frame rate: %s (fps)' % self.frameRate)
        print('                Codec: %s' % self.codec)
        
        printOK = 1
        try:
            f = self.numFrames
        except AttributeError:
            printOK = 0
            
        if printOK:
            print('           Num frames: %s' % self.numFrames)
            print('               Height: %s' % self.height)
            print('                Width: %s' % self.height)
            print('             Detector: %s' % self.detector)
            print('            Extractor: %s' % self.extractor)
            
    def printROIInfo(self):
        print('      ROI type: ' + self.typeROI)
        if self.typeROI is 'rect':
            print('   Rect coords: ' + str(self.rectCoords))
        elif self.typeROI is 'skin_fix':
            print('   Skin thresh: ' + str(self.skinThresh_fix))
        elif self.typeROI is 'skin_adapt':
            print('   Skin thresh: ' + str(self.skinThresh_adapt))

    def showVideo(self):
        from ipywidgets import interact
        import ipywidgets as widgets

        n = self.numFrames
        def view_image(frame):
            
            idx = frame-1

            if self.processedFaces.size == 0:
                face = self.faces[idx]
            else:
                face = self.processedFaces[idx]

            if self.typeROI is 'rect':
                plt.imshow(face, interpolation='nearest')
                
                ax = plt.gca()                
        
                for coord in self.rectCoords:
                    rect = patches.Rectangle((coord[0],coord[1]),
                            coord[2],coord[3],linewidth=1,edgecolor='y',facecolor='none')
                    ax.add_patch(rect)

            elif self.typeROI is 'skin_fix':
                lower = np.array([0, self.skinThresh_fix[0], self.skinThresh_fix[1]], dtype = "uint8")
                upper = np.array([20, 255, 255], dtype = "uint8")
                converted = cv2.cvtColor(face, cv2.COLOR_RGB2HSV)
                skinMask = cv2.inRange(converted, lower, upper)
                skinFace = cv2.bitwise_and(face, face, mask=skinMask)                
                plt.imshow(skinFace, interpolation='nearest')
                
            elif self.typeROI is 'skin_adapt':     
                sd = SkinDetect(strength=self.skinThresh_adapt)
                sd.compute_stats(face)
                skinFace = sd.get_skin(face, filt_kern_size=7, verbose=False, plot=False)     
                plt.imshow(skinFace, interpolation='nearest')

        interact(view_image, frame=widgets.IntSlider(min=1, max=n, step=1, value=1))

    def __opencvRead(self):
        vid = cv2.VideoCapture(self.filename)
        frames = []
        retval, frame = vid.read()
        while retval == True:
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            retval, frame = vid.read()
        vid.release()
        return np.asarray(frames)

    def __extractRectSignal(self, frameSubset):
        """ Extract R,G,B values on all ROIs of a frame subset """
        
        assert self.processedFaces.size > 0, "Faces are not processed yet! Please call runOffline first"

        self.faceSignal = []

        i = 0
        for r in frameSubset:
            face = self.processedFaces[r]
            H = face.shape[0]
            W = face.shape[1]
            
            # take frame-level rois
            rois = []
            for roi in self.rectCoords:
                x = roi[0]
                y = roi[1]
                w = min(x + roi[2], W)
                h = min(y + roi[3], H)
                rois.append(face[y:h,x:w,:])

            # take all rois of the frame
            self.faceSignal.append(rois)
            i += 1

    def __extractSkinSignal(self, frameSubset, count=None, frameByframe=False):
        """ Extract R,G,B values from skin-based roi of a frame subset """
        
        assert self.processedFaces.size > 0, "Faces are not processed yet! Please call runOffline first"

        self.faceSignal = []
        
        cp = self.cropSize
        skinFace = np.zeros([cp[0],cp[1],3], dtype='uint8')

        # -- loop on frames
        for i,r in enumerate(frameSubset):
            face = self.processedFaces[r]

            if self.typeROI == 'skin_fix':        
                assert len(self.skinThresh_fix) == 2, "Please provide 2 values for Fixed Skin Detector"           
                lower = np.array([0, self.skinThresh_fix[0], self.skinThresh_fix[1]], dtype = "uint8")
                upper = np.array([20, 255, 255], dtype = "uint8")
                converted = cv2.cvtColor(face, cv2.COLOR_RGB2HSV)
                skinMask = cv2.inRange(converted, lower, upper)
                skinFace = cv2.bitwise_and(face, face, mask=skinMask)
                self.faceSignal.append([skinFace])
        
            elif self.typeROI == 'skin_adapt':  
                if count == 0 and i == 0:
                    self.sd = SkinDetect(strength=self.skinThresh_adapt)
                    self.sd.compute_stats(face)
                
                if frameByframe and i > 0:
                    self.sd.compute_stats(face)
                
                skinFace = self.sd.get_skin(face, filt_kern_size=0, verbose=False, plot=False)

                self.faceSignal.append([skinFace])

    def __extractFace(self, video, method, t_downsample_rate=2):

        # -- save on GPU
        #self.facesGPU = cp.asarray(self.faces)  # move the data to the current device.

        if method == 'dlib':
            # -- dlib detector
            detector = dlib.get_frontal_face_detector()
            if os.path.exists("resources/shape_predictor_68_face_landmarks.dat"):
                file_predict = "resources/shape_predictor_68_face_landmarks.dat"
            elif os.path.exists("../resources/shape_predictor_68_face_landmarks.dat"):
                file_predict = "../resources/shape_predictor_68_face_landmarks.dat"
            predictor = dlib.shape_predictor(file_predict)
            self.faces = np.zeros([self.numFrames, self.cropSize[0], self.cropSize[1], 3],
                                  dtype='uint8')

            # -- loop on frames
            cp = self.cropSize
            self.faces = np.zeros([self.numFrames,cp[0],cp[1],3], dtype='uint8')
            for i in range(self.numFrames):
                frame = video[i,:,:,:]
                # -- Detect face using dlib
                self.numFaces = 0
                facesRect = detector(frame, 0)
                if len(facesRect) > 0:
                    # -- process only the first face
                    self.numFaces += 1
                    rect = facesRect[0]
                    x0 = rect.left()
                    y0 = rect.top()
                    w = rect.width()
                    h = rect.height()

                    # -- extract cropped faces
                    shape = predictor(frame, rect)
                    f = dlib.get_face_chip(frame, shape, size=self.cropSize[0], padding=self.facePadding)
                    self.faces[i,:,:,:] = f.astype('uint8')

                if self.verb: printutils.printProgressBar(i, self.numFrames, prefix = 'Processing:', suffix = 'Complete', length = 50)

                else:
                    print("No face detected at frame %s",i)

        elif method == 'mtcnn_kalman':
            #mtcnn detector
            from mtcnn import MTCNN
            detector = MTCNN()

            h0 = None
            w0 = None
            crop = np.zeros([2,2,2])
            skipped_frames = 0
            
            while crop.shape[:2] != (h0,w0):
                if skipped_frames > 0:
                    print("\nWARNING! Strange Face Crop... Skipping frame " + str(skipped_frames) + '...')
                frame = video[skipped_frames,:,:,:]
                detection = detector.detect_faces(frame)
                
                if len(detection) > 1:
                    areas = []
                    for det in detection:
                        areas.append(det['box'][2] * det['box'][3])
                    areas = np.array(areas)
                    ia = np.argsort(areas)
                    [x0, y0, w0, h0] = detection[ia[-1]]['box']
                else:
                    [x0, y0, w0, h0] = detection[0]['box']

                w0 = 2*(int(w0/2))
                h0 = 2*(int(h0/2))
                #Cropping face
                crop = frame[y0:y0+h0, x0:x0+w0, :]

                skipped_frames += 1

            self.cropSize = crop.shape[:2]

            if skipped_frames > 1:
                self.numFrames = self.numFrames - skipped_frames
                new_time_vid_start = skipped_frames / self.frameRate

                if new_time_vid_start > self.time_vid_start:
                    self.time_vid_start = new_time_vid_start
                    print("\tVideo now starts at " + str(self.time_vid_start) + " seconds\n")
            
            self.faces = np.zeros([self.numFrames, self.cropSize[0], self.cropSize[1], 3], dtype='uint8')
            self.faces[0,:,:,:] = crop

            #set the initial tracking window
            state = np.array([int(x0+w0/2),int(y0+h0/2),0,0], dtype='float64') # initial position

            #Setting up Kalman Filter
            kalman = cv2.KalmanFilter(4,2,0)
            kalman.transitionMatrix = np.array([[1., 0., .1, 0.],
                                                [0., 1., 0., .1],
                                                [0., 0., 1., 0.],
                                                [0., 0., 0., 1.]])
            kalman.measurementMatrix = 1. * np.eye(2, 4)
            kalman.processNoiseCov = 1e-5 * np.eye(4, 4)
            kalman.measurementNoiseCov = 1e-3 * np.eye(2, 2)
            kalman.errorCovPost = 1e-1 * np.eye(4, 4)
            kalman.statePost = state
            measurement = np.array([int(x0+w0/2), int(y0+h0/2)], dtype='float64')

            for i in range(skipped_frames,self.numFrames):
                frame = video[i,:,:,:]

                if i%t_downsample_rate == 0:
                    detection = detector.detect_faces(frame)
                    if len(detection) != 0:
                        areas = []
                        if len(detection) > 1:
                            for det in detection:
                                areas.append(det['box'][2] * det['box'][3])
                            areas = np.array(areas)
                            ia = np.argsort(areas)

                            [x0, y0, w, h] = detection[ia[-1]]['box']
                        else:
                            [x0, y0, w, h] = detection[0]['box']
                        
                        not_found = False
                    else:
                        not_found = True

                prediction = kalman.predict() #prediction

                if i%t_downsample_rate == 0 and not not_found:
                    measurement = np.array([x0+w/2, y0+h/2], dtype='float64')
                    posterior = kalman.correct(measurement)
                    [cx0, cy0, wn, hn] = posterior.astype(int)
                else:
                    [cx0, cy0, wn, hn] = prediction.astype(int)

                #Cropping with new bounding box
                crop = frame[int(cy0-h0/2):int(cy0+h0/2), int(cx0-w0/2):int(cx0+w0/2), :]

                if crop.shape[:2] != self.faces.shape[1:3]:
                    print("WARNING! Strange face crop: video frame " + str(i) +" probably does not contain the whole face... Reshaping Crop\n")
                    crop = cv2.resize(crop, (self.faces.shape[2], self.faces.shape[1]))

                self.faces[i,:,:,:] = crop.astype('uint8')

        elif method == 'mtcnn':
            #mtcnn detector
            from mtcnn import MTCNN
            #from utils.FaceAligner import FaceAligner
            detector = MTCNN()

            h0 = None
            w0 = None
            crop = np.zeros([2,2,2])
            skipped_frames = 0
            
            while crop.shape[:2] != (h0,w0):
                if skipped_frames > 0:
                    print("\nWARNING! Strange Face Crop... Skipping frame " + str(skipped_frames) + '...')
                frame = video[skipped_frames,:,:,:]
                detection = detector.detect_faces(frame)

                if len(detection) == 0:
                    skipped_frames += 1
                    continue
                
                if len(detection) > 1:
                    areas = []
                    for det in detection:
                        areas.append(det['box'][2] * det['box'][3])
                    areas = np.array(areas)
                    ia = np.argsort(areas)
                    [x0, y0, w0, h0] = detection[ia[-1]]['box']
                    nose = detection[ia[-1]]['keypoints']['nose']
                    r_eye = detection[ia[-1]]['keypoints']['right_eye']
                    l_eye = detection[ia[-1]]['keypoints']['left_eye']
                else:
                    [x0, y0, w0, h0] = detection[0]['box']
                    nose = detection[0]['keypoints']['nose']
                    r_eye = detection[0]['keypoints']['right_eye']
                    l_eye = detection[0]['keypoints']['left_eye']

                w0 = 2*(int(w0/2))
                h0 = 2*(int(h0/2))
                barycenter = (np.array(nose) + np.array(r_eye) + np.array(l_eye)) / 3.
                cy0 = barycenter[1]
                cx0 = barycenter[0]
                #Cropping face
                crop = frame[int(cy0-h0/2):int(cy0+h0/2), int(cx0-w0/2):int(cx0+w0/2), :]

                skipped_frames += 1

            #fa = FaceAligner(desiredLeftEye=(0.3, 0.3),desiredFaceWidth=w0, desiredFaceHeight=h0)
            #crop_align = fa.align(frame, r_eye, l_eye)

            self.cropSize = crop.shape[:2]

            if skipped_frames > 1:
                self.numFrames = self.numFrames - skipped_frames
                new_time_vid_start = skipped_frames / self.frameRate

                if new_time_vid_start > self.time_vid_start:
                    self.time_vid_start = new_time_vid_start
                    print("\tVideo now starts at " + str(self.time_vid_start) + " seconds\n")
            
            self.faces = np.zeros([self.numFrames, self.cropSize[0], self.cropSize[1], 3], dtype='uint8')
            self.faces[0,:,:,:] = crop

            old_detection = detection
            for i in range(skipped_frames,self.numFrames):
                frame = video[i,:,:,:]

                new_detection = detector.detect_faces(frame)
                areas = []

                if len(new_detection) == 0:
                    new_detection = old_detection
            
                if len(new_detection) > 1:
                    for det in new_detection:
                        areas.append(det['box'][2] * det['box'][3])
                    areas = np.array(areas)
                    ia = np.argsort(areas)

                    [x0, y0, w, h] = new_detection[ia[-1]]['box']
                    nose = new_detection[ia[-1]]['keypoints']['nose']
                    r_eye = new_detection[ia[-1]]['keypoints']['right_eye']
                    l_eye = new_detection[ia[-1]]['keypoints']['left_eye']
                else:
                    [x0, y0, w, h] = new_detection[0]['box']
                    nose = new_detection[0]['keypoints']['nose']
                    r_eye = new_detection[0]['keypoints']['right_eye']
                    l_eye = new_detection[0]['keypoints']['left_eye']
                    
                barycenter = (np.array(nose) + np.array(r_eye) + np.array(l_eye)) / 3.
                cy0 = barycenter[1]
                cx0 = barycenter[0]
                #Cropping with new bounding box
                crop = frame[int(cy0-h0/2):int(cy0+h0/2), int(cx0-w0/2):int(cx0+w0/2), :]

                if crop.shape[:2] != self.faces.shape[1:3]:
                    print("WARNING! Strange face crop: video frame " + str(i) +" probably does not contain the whole face... Reshaping Crop\n")
                    crop = cv2.resize(crop, (self.faces.shape[2], self.faces.shape[1]))

                self.faces[i,:,:,:] = crop.astype('uint8')
                old_detection = new_detection
        
                if self.verb: printutils.printProgressBar(i, self.numFrames, prefix = 'Processing:', suffix = 'Complete', length = 50)
        else:

            raise ValueError('Unrecognized Face detection method. Please use "dlib" or "mtcnn"')

    def __rectRegions2Coord(self, rectRegions):
      
        # regions 'forehead'
        #         'lcheek'
        #         'rcheek'
        #         'nose'
        assert len(self.faces) > 0, "Faces not found, please run getCroppedFaces first!"

        w = self.faces[0].shape[1]
        h = self.faces[0].shape[0]
        
        coords = []

        for roi in rectRegions:
            print(self.detector)
            if roi is 'forehead':
                if self.detector == 'dlib':
                    x_f = int(w * .34)
                    y_f = int(h * .05)
                    w_f = int(w * .32)
                    h_f = int(h * .05)

                elif (self.detector == 'mtcnn') or (self.detector == 'mtcnn_kalman'):
                    x_f = int(w * .20)
                    y_f = int(h * .10)
                    w_f = int(w * .60)
                    h_f = int(h * .12)

                coords.append([x_f, y_f, w_f, h_f])

            elif roi is 'lcheek':
                if self.detector == 'dlib':
                    x_c = int(w * .22)
                    y_c = int(h * .40)
                    w_c = int(w * .14)
                    h_c = int(h * .11)

                elif (self.detector == 'mtcnn') or (self.detector == 'mtcnn_kalman'):
                    x_c = int(w * .15)
                    y_c = int(h * .54)
                    w_c = int(w * .15)
                    h_c = int(h * .11)

                coords.append([x_c, y_c, w_c, h_c])

            elif roi is 'rcheek':
                if self.detector == 'dlib':
                    x_c = int(w * .64)
                    y_c = int(h * .40)
                    w_c = int(w * .14)
                    h_c = int(h * .11)

                elif (self.detector == 'mtcnn') or (self.detector == 'mtcnn_kalman'):
                    x_c = int(w * .70)
                    y_c = int(h * .54)
                    w_c = int(w * .15)
                    h_c = int(h * .11)

                coords.append([x_c, y_c, w_c, h_c])

            elif roi is 'nose':
                if self.detector == 'dlib':
                    x_c = int(w * .40)
                    y_c = int(h * .35)
                    w_c = int(w * .20)
                    h_c = int(h * .05)

                elif (self.detector == 'mtcnn') or (self.detector == 'mtcnn_kalman'):
                    x_c = int(w * .35)
                    y_c = int(h * .50)
                    w_c = int(w * .30)
                    h_c = int(h * .08)

                coords.append([x_c, y_c, w_c, h_c])

            else:
                raise ValueError('Unrecognized rect region name.')

        return coords

    def __sort_nicely(self, l): 
        """ Sort the given list in the way that humans expect. 
        """ 
        convert = lambda text: int(text) if text.isdigit() else text 
        alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ] 
        l.sort( key=alphanum_key )
        return l

    def __loadFrames(self):
        
        # -- delete the compressed if exists
        cmpFile = os.path.join(self.path,self.filenameCompressed) 
        if os.path.exists(cmpFile):
            os.remove(cmpFile)
        
        # -- get filenames within dir
        f_names = self.__sort_nicely(os.listdir(self.path))
        frames = []
        for n in range(len(f_names)):
            filename = os.path.join(self.path,f_names[n])    
            frames.append(cv2.imread(filename)[:,:,::-1])
        
        frames = np.array(frames)
        return frames