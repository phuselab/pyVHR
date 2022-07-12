import cv2
import mediapipe as mp
import numpy as np
from pyVHR.extraction.utils import *
from pyVHR.extraction.skin_extraction_methods import *
from pyVHR.extraction.sig_extraction_methods import *
from pyVHR.utils.cuda_utils import *

"""
This module defines classes or methods used for Signal extraction and processing.
"""


class SignalProcessing():
    """
        This class performs offline signal extraction with different methods:

        - holistic.

        - squared / rectangular patches.
    """

    def __init__(self):
        # Common parameters #
        self.tot_frames = None
        self.visualize_skin_collection = []
        self.skin_extractor = SkinExtractionConvexHull('CPU')
        # Patches parameters #
        high_prio_ldmk_id, mid_prio_ldmk_id = get_magic_landmarks()
        self.ldmks = high_prio_ldmk_id + mid_prio_ldmk_id
        self.square = None
        self.rects = None
        self.visualize_skin = False
        self.visualize_landmarks = False
        self.visualize_landmarks_number = False
        self.visualize_patch = False
        self.font_size = 0.3
        self.font_color = (255, 0, 0, 255)
        self.visualize_skin_collection = []
        self.visualize_landmarks_collection = []

    def choose_cuda_device(self, n):
        """
        Choose a CUDA device.

        Args:  
            n (int): number of a CUDA device.

        """
        select_cuda_device(n)

    def display_cuda_device(self):
        """
        Display your CUDA devices.
        """
        cuda_info()

    def set_total_frames(self, n):
        """
        Set the total frames to be processed; if you want to process all the possible frames use n = 0.
        
        Args:  
            n (int): number of frames to be processed.
            
        """
        if n < 0:
            print("[ERROR] n must be a positive number!")
        self.tot_frames = int(n)

    def set_skin_extractor(self, extractor):
        """
        Set the skin extractor that will be used for skin extraction.
        
        Args:  
            extractor: instance of a skin_extraction class (see :py:mod:`pyVHR.extraction.skin_extraction_methods`).
            
        """
        self.skin_extractor = extractor

    def set_visualize_skin_and_landmarks(self, visualize_skin=False, visualize_landmarks=False, visualize_landmarks_number=False, visualize_patch=False):
        """
        Set visualization parameters. You can retrieve visualization output with the 
        methods :py:meth:`pyVHR.extraction.sig_processing.SignalProcessing.get_visualize_skin` 
        and :py:meth:`pyVHR.extraction.sig_processing.SignalProcessing.get_visualize_patches`

        Args:  
            visualize_skin (bool): The skin and the patches will be visualized.
            visualize_landmarks (bool): The landmarks (centers of patches) will be visualized.
            visualize_landmarks_number (bool): The landmarks number will be visualized.
            visualize_patch (bool): The patches outline will be visualized.
        
        """
        self.visualize_skin = visualize_skin
        self.visualize_landmarks = visualize_landmarks
        self.visualize_landmarks_number = visualize_landmarks_number
        self.visualize_patch = visualize_patch

    def get_visualize_skin(self):
        """
        Get the skin images produced by the last processing. Remember to 
        set :py:meth:`pyVHR.extraction.sig_processing.SignalProcessing.set_visualize_skin_and_landmarks`
        correctly.
        
        Returns:
            list of ndarray: list of cv2 images; each image is a ndarray with shape [rows, columns, rgb_channels].
        """
        return self.visualize_skin_collection

    def get_visualize_patches(self):
        """
        Get the 'skin+patches' images produced by the last processing. Remember to 
        set :py:meth:`pyVHR.extraction.sig_processing.SignalProcessing.set_visualize_skin_and_landmarks`
        correctly.
        
        Returns:
            list of ndarray: list of cv2 images; each image is a ndarray with shape [rows, columns, rgb_channels].
        """
        return self.visualize_landmarks_collection

    def extract_raw(self, videoFileName):
        """
        Extracts raw frames from video.

        Args:
            videoFileName (str): video file name or path.

        Returns: 
            ndarray: raw frames with shape [num_frames, height, width, rgb_channels].
        """

        frames = []
        for frame in extract_frames_yield(videoFileName):
                frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))   # convert to RGB

        return np.array(frames)



    ### HOLISTIC METHODS ###

    def extract_raw_holistic(self, videoFileName):
        """
        Locates the skin pixels in each frame. This method is intended for rPPG methods that use raw video signal.

        Args:
            videoFileName (str): video file name or path.

        Returns: 
            float32 ndarray: raw signal as float32 ndarray with shape [num_frames, rows, columns, rgb_channels].
        """

        skin_ex = self.skin_extractor

        mp_drawing = mp.solutions.drawing_utils
        mp_face_mesh = mp.solutions.face_mesh
        PRESENCE_THRESHOLD = 0.5
        VISIBILITY_THRESHOLD = 0.5

        sig = []
        processed_frames_count = 0

        with mp_face_mesh.FaceMesh(
                max_num_faces=1,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5) as face_mesh:
            for frame in extract_frames_yield(videoFileName):
                # convert the BGR image to RGB.
                image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                processed_frames_count += 1
                width = image.shape[1]
                height = image.shape[0]
                # [landmarks, info], with info->x_center ,y_center, r, g, b
                ldmks = np.zeros((468, 5), dtype=np.float32)
                ldmks[:, 0] = -1.0
                ldmks[:, 1] = -1.0
                ### face landmarks ###
                results = face_mesh.process(image)
                if results.multi_face_landmarks:
                    face_landmarks = results.multi_face_landmarks[0]
                    landmarks = [l for l in face_landmarks.landmark]
                    for idx in range(len(landmarks)):
                        landmark = landmarks[idx]
                        if not ((landmark.HasField('visibility') and landmark.visibility < VISIBILITY_THRESHOLD)
                                or (landmark.HasField('presence') and landmark.presence < PRESENCE_THRESHOLD)):
                            coords = mp_drawing._normalized_to_pixel_coordinates(
                                landmark.x, landmark.y, width, height)
                            if coords:
                                ldmks[idx, 0] = coords[1]
                                ldmks[idx, 1] = coords[0]
                    ### skin extraction ###
                    cropped_skin_im, full_skin_im = skin_ex.extract_skin(
                        image, ldmks)
                else:
                    cropped_skin_im = np.zeros_like(image)
                    full_skin_im = np.zeros_like(image)
                if self.visualize_skin == True:
                    self.visualize_skin_collection.append(full_skin_im)
                ### sig computing ###
                sig.append(full_skin_im)
                ### loop break ###
                if self.tot_frames is not None and self.tot_frames > 0 and processed_frames_count >= self.tot_frames:
                    break
        sig = np.array(sig, dtype=np.float32)
        return sig

    def extract_holistic(self, videoFileName):
        """
        This method compute the RGB-mean signal using the whole skin (holistic);

        Args:
            videoFileName (str): video file name or path.

        Returns: 
            float32 ndarray: RGB signal as ndarray with shape [num_frames, 1, rgb_channels]. The second dimension is 1 because
            the whole skin is considered as one estimators.
        """
        self.visualize_skin_collection = []

        skin_ex = self.skin_extractor

        mp_drawing = mp.solutions.drawing_utils
        mp_face_mesh = mp.solutions.face_mesh
        PRESENCE_THRESHOLD = 0.5
        VISIBILITY_THRESHOLD = 0.5

        sig = []
        processed_frames_count = 0

        with mp_face_mesh.FaceMesh(
                max_num_faces=1,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5) as face_mesh:
            for frame in extract_frames_yield(videoFileName):
                # convert the BGR image to RGB.
                image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                processed_frames_count += 1
                width = image.shape[1]
                height = image.shape[0]
                # [landmarks, info], with info->x_center ,y_center, r, g, b
                ldmks = np.zeros((468, 5), dtype=np.float32)
                ldmks[:, 0] = -1.0
                ldmks[:, 1] = -1.0
                ### face landmarks ###
                results = face_mesh.process(image)
                if results.multi_face_landmarks:
                    face_landmarks = results.multi_face_landmarks[0]
                    landmarks = [l for l in face_landmarks.landmark]
                    for idx in range(len(landmarks)):
                        landmark = landmarks[idx]
                        if not ((landmark.HasField('visibility') and landmark.visibility < VISIBILITY_THRESHOLD)
                                or (landmark.HasField('presence') and landmark.presence < PRESENCE_THRESHOLD)):
                            coords = mp_drawing._normalized_to_pixel_coordinates(
                                landmark.x, landmark.y, width, height)
                            if coords:
                                ldmks[idx, 0] = coords[1]
                                ldmks[idx, 1] = coords[0]
                    ### skin extraction ###
                    cropped_skin_im, full_skin_im = skin_ex.extract_skin(
                        image, ldmks)
                else:
                    cropped_skin_im = np.zeros_like(image)
                    full_skin_im = np.zeros_like(image)
                if self.visualize_skin == True:
                    self.visualize_skin_collection.append(full_skin_im)
                ### sig computing ###
                sig.append(holistic_mean(
                    cropped_skin_im, np.int32(SignalProcessingParams.RGB_LOW_TH), np.int32(SignalProcessingParams.RGB_HIGH_TH)))
                ### loop break ###
                if self.tot_frames is not None and self.tot_frames > 0 and processed_frames_count >= self.tot_frames:
                    break
        sig = np.array(sig, dtype=np.float32)
        return sig

    ### PATCHES METHODS ###

    def set_landmarks(self, landmarks_list):
        """
        Set the patches centers (landmarks) that will be used for signal processing. There are 468 facial points you can
        choose; for visualizing their identification number please use :py:meth:`pyVHR.plot.visualize.visualize_landmarks_list`.

        Args:
            landmarks_list (list): list of positive integers between 0 and 467 that identify patches centers (landmarks).
        """
        if not isinstance(landmarks_list, list):
            print("[ERROR] landmarks_set must be a list!")
            return
        self.ldmks = landmarks_list

    def set_square_patches_side(self, square_side):
        """
        Set the dimension of the square patches that will be used for signal processing. There are 468 facial points you can
        choose; for visualizing their identification number please use :py:meth:`pyVHR.plot.visualize.visualize_landmarks_list`.

        Args:
            square_side (float): positive float that defines the length of the square patches.
        """
        if not isinstance(square_side, float) or square_side <= 0.0:
            print("[ERROR] square_side must be a positive float!")
            return
        self.square = square_side

    def set_rect_patches_sides(self, rects_dim):
        """
        Set the dimension of each rectangular patch. There are 468 facial points you can
        choose; for visualizing their identification number please use :py:meth:`pyVHR.plot.visualize.visualize_landmarks_list`.

        Args:
            rects_dim (float32 ndarray): positive float32 np.ndarray of shape [num_landmarks, 2]. If the list of used landmarks is [1,2,3] 
                and rects_dim is [[10,20],[12,13],[40,40]] then the landmark number 2 will have a rectangular patch of xy-dimension 12x13.
        """
        if type(rects_dim) != type(np.array([])):
            print("[ERROR] rects_dim must be an np.ndarray!")
            return
        if rects_dim.shape[0] != len(self.ldmks) and rects_dim.shape[1] != 2:
            print("[ERROR] incorrect rects_dim shape!")
            return
        self.rects = rects_dim

    def extract_patches(self, videoFileName, region_type, sig_extraction_method):
        """
        This method compute the RGB-mean signal using specific skin regions (patches).

        Args:
            videoFileName (str): video file name or path.
            region_type (str): patches types can be  "squares" or "rects".
            sig_extraction_method (str): RGB signal can be computed with "mean" or "median". We recommend to use mean.

        Returns: 
            float32 ndarray: RGB signal as ndarray with shape [num_frames, num_patches, rgb_channels].
        """
        if self.square is None and self.rects is None:
            print(
                "[ERROR] Use set_landmarks_squares or set_landmarkds_rects before calling this function!")
            return None
        if region_type != "squares" and region_type != "rects":
            print("[ERROR] Invalid landmarks region type!")
            return None
        if sig_extraction_method != "mean" and sig_extraction_method != "median":
            print("[ERROR] Invalid signal extraction method!")
            return None

        ldmks_regions = None
        if region_type == "squares":
            ldmks_regions = np.float32(self.square)
        elif region_type == "rects":
            ldmks_regions = np.float32(self.rects)

        sig_ext_met = None
        if sig_extraction_method == "mean":
            if region_type == "squares":
                sig_ext_met = landmarks_mean
            elif region_type == "rects":
                sig_ext_met = landmarks_mean_custom_rect
        elif sig_extraction_method == "median":
            if region_type == "squares":
                sig_ext_met = landmarks_median
            elif region_type == "rects":
                sig_ext_met = landmarks_median_custom_rect

        self.visualize_skin_collection = []
        self.visualize_landmarks_collection = []

        skin_ex = self.skin_extractor

        mp_drawing = mp.solutions.drawing_utils
        mp_face_mesh = mp.solutions.face_mesh
        PRESENCE_THRESHOLD = 0.5
        VISIBILITY_THRESHOLD = 0.5

        sig = []
        processed_frames_count = 0
        self.patch_landmarks = []
        self.cropped_skin_im_shapes = [[], []]
        with mp_face_mesh.FaceMesh(
                max_num_faces=1,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5) as face_mesh:
            for frame in extract_frames_yield(videoFileName):
                # convert the BGR image to RGB.
                image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                processed_frames_count += 1
                width = image.shape[1]
                height = image.shape[0]
                # [landmarks, info], with info->x_center ,y_center, r, g, b
                ldmks = np.zeros((468, 5), dtype=np.float32)
                ldmks[:, 0] = -1.0
                ldmks[:, 1] = -1.0
                magic_ldmks = []
                ### face landmarks ###
                results = face_mesh.process(image)
                if results.multi_face_landmarks:
                    face_landmarks = results.multi_face_landmarks[0]
                    landmarks = [l for l in face_landmarks.landmark]
                    for idx in range(len(landmarks)):
                        landmark = landmarks[idx]
                        if not ((landmark.HasField('visibility') and landmark.visibility < VISIBILITY_THRESHOLD)
                                or (landmark.HasField('presence') and landmark.presence < PRESENCE_THRESHOLD)):
                            coords = mp_drawing._normalized_to_pixel_coordinates(
                                landmark.x, landmark.y, width, height)
                            if coords:
                                ldmks[idx, 0] = coords[1]
                                ldmks[idx, 1] = coords[0]
                                
                    ### skin extraction ###
                    cropped_skin_im, full_skin_im = skin_ex.extract_skin(image, ldmks)

                    self.cropped_skin_im_shapes[0].append(cropped_skin_im.shape[0])
                    self.cropped_skin_im_shapes[1].append(cropped_skin_im.shape[1])

                else:
                    cropped_skin_im = np.zeros_like(image)
                    full_skin_im = np.zeros_like(image)
                    self.cropped_skin_im_shapes[1].append(cropped_skin_im.shape[1])

                ### sig computing ###
                for idx in self.ldmks:
                    magic_ldmks.append(ldmks[idx])
                magic_ldmks = np.array(magic_ldmks, dtype=np.float32)
                temp = sig_ext_met(magic_ldmks, full_skin_im, ldmks_regions,
                                   np.int32(SignalProcessingParams.RGB_LOW_TH), 
                                   np.int32(SignalProcessingParams.RGB_HIGH_TH))
                sig.append(temp)

                # save landmarks coordinates
                self.patch_landmarks.append(magic_ldmks[:,0:3])

                # visualize patches and skin
                if self.visualize_skin == True:
                    self.visualize_skin_collection.append(full_skin_im)
                if self.visualize_landmarks == True:
                    annotated_image = full_skin_im.copy()
                    color = np.array([self.font_color[0],
                                      self.font_color[1], self.font_color[2]], dtype=np.uint8)
                    for idx in self.ldmks:
                        cv2.circle(
                            annotated_image, (int(ldmks[idx, 1]), int(ldmks[idx, 0])), radius=0, color=self.font_color, thickness=-1)
                        if self.visualize_landmarks_number == True:
                            cv2.putText(annotated_image, str(idx),
                                        (int(ldmks[idx, 1]), int(ldmks[idx, 0])), cv2.FONT_HERSHEY_SIMPLEX, self.font_size,  self.font_color,  1)
                    if self.visualize_patch == True:
                        if region_type == "squares":
                            sides = np.array([self.square] * len(magic_ldmks))
                            annotated_image = draw_rects(
                                annotated_image, np.array(magic_ldmks[:, 1]), np.array(magic_ldmks[:, 0]), sides, sides, color)
                        elif region_type == "rects":
                            annotated_image = draw_rects(
                                annotated_image, np.array(magic_ldmks[:, 1]), np.array(magic_ldmks[:, 0]), np.array(self.rects[:, 0]), np.array(self.rects[:, 1]), color)
                    self.visualize_landmarks_collection.append(
                        annotated_image)
                ### loop break ###
                if self.tot_frames is not None and self.tot_frames > 0 and processed_frames_count >= self.tot_frames:
                    break
        sig = np.array(sig, dtype=np.float32)
        return np.copy(sig[:, :, 2:])

    def get_landmarks(self):
        """
        Returns landmarks ndarray with shape [num_frames, num_estimators, 2-coords] or empty array 
        """
        if hasattr(self, 'patch_landmarks'):
            return np.array(self.patch_landmarks)
        else:
            return np.empty(0)

    def get_cropped_skin_im_shapes(self):
        """
        Returns cropped skin shapes with shape [height, width, rgb] or empty array
        """
        if hasattr(self, "cropped_skin_im_shapes"):
            return np.array(self.cropped_skin_im_shapes)
        else:
            return np.empty((0, 0, 0))


