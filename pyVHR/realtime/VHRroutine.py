from threading import Thread
import time
from time import sleep
from pyVHR.realtime.params import Params
from pyVHR.realtime.video_capture import VideoCapture
import cv2
import numpy as np
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
import PySimpleGUI as sg
import numpy as np
import plotly.graph_objects as go
import queue
import statistics


class SharedData:
    def __init__(self):
        self.q_bpm = queue.Queue()
        self.q_video_image = queue.Queue()
        self.q_skin_image = queue.Queue()
        self.q_patches_image = queue.Queue()
        self.q_stop = queue.Queue()
        self.q_stop_cap = queue.Queue()
        self.q_frames = queue.Queue()


def VHRroutine(sharedData):
    sig_ext_met = None
    ldmks_regions = None
    # Holistic settings #
    if Params.approach == 'holistic':
        sig_ext_met = holistic_mean
    # Patches settings #
    elif Params.approach == 'patches':
        # extraction method
        if Params.type == "mean" and Params.patches == "squares":
            sig_ext_met = landmarks_mean
        elif Params.type == "mean" and Params.patches == "rects":
            sig_ext_met = landmarks_mean_custom_rect
        elif Params.type == "median" and Params.patches == "squares":
            sig_ext_met = landmarks_median
        elif Params.type == "median" and Params.patches == "rects":
            sig_ext_met = landmarks_median_custom_rect
        # patches dims
        if Params.patches == "squares":
            ldmks_regions = np.float32(Params.squares_dim)
        elif Params.patches == "rects":
            ldmks_regions = np.float32(Params.rects_dims)

    SignalProcessingParams.RGB_LOW_TH = np.int32(
        Params.sig_color_low_threshold)
    SignalProcessingParams.RGB_HIGH_TH = np.int32(
        Params.sig_color_high_threshold)
    SkinProcessingParams.RGB_LOW_TH = np.int32(
        Params.skin_color_low_threshold)
    SkinProcessingParams.RGB_HIGH_TH = np.int32(
        Params.skin_color_high_threshold)

    color = np.array([Params.font_color[0],
                      Params.font_color[1], Params.font_color[2]], dtype=np.uint8)

    skin_ex = None
    if Params.skin_extractor == 'convexhull':
        skin_ex = SkinExtractionConvexHull()
    elif Params.skin_extractor == 'faceparsing':
        skin_ex = SkinExtractionFaceParsing()

    mp_drawing = mp.solutions.drawing_utils
    mp_face_mesh = mp.solutions.face_mesh
    PRESENCE_THRESHOLD = 0.5
    VISIBILITY_THRESHOLD = 0.5

    if Params.fps_fixed is not None:
        fps = Params.fps_fixed
    else:
        fps = get_fps(Params.videoFileName)
    tot_frames = int(Params.tot_sec*fps)

    sig = []
    processed_frames_count = 0
    sig_buff_dim = int(fps * Params.winSize)
    sig_stride = int(fps * Params.stride)
    sig_buff_counter = sig_stride

    BPM_obj = None

    timeCount = []

    cap = VideoCapture(Params.videoFileName, sharedData, fps=fps,
                       sleep=Params.fake_delay, resize=Params.resize)

    send_images_count = 0
    send_images_stride = 3

    with mp_face_mesh.FaceMesh(
            max_num_faces=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5) as face_mesh:
        while True:
            start_time = time.perf_counter()*1000
            frame = None
            if not sharedData.q_frames.empty():  # read frames from shared data
                frame = sharedData.q_frames.get()
                if type(frame) == int:  # cap stopped
                    break
            if not sharedData.q_stop.empty():  # GUI stopped
                sharedData.q_stop.get()
                break
            if frame is None:
                continue
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
                cropped_skin_im, full_skin_im = skin_ex.extract_skin(
                    image, ldmks)
            else:
                cropped_skin_im = np.zeros_like(image)
                full_skin_im = np.zeros_like(image)
            ### SIG ###
            if Params.approach == 'patches':
                magic_ldmks = np.array(
                    ldmks[Params.landmarks_list], dtype=np.float32)
                temp = sig_ext_met(magic_ldmks, full_skin_im, ldmks_regions,
                                   np.int32(SignalProcessingParams.RGB_LOW_TH), np.int32(SignalProcessingParams.RGB_HIGH_TH))
                temp = temp[:, 2:]  # keep only rgb mean
            elif Params.approach == 'holistic':
                temp = sig_ext_met(cropped_skin_im, np.int32(
                    SignalProcessingParams.RGB_LOW_TH), np.int32(SignalProcessingParams.RGB_HIGH_TH))
            sig.append(temp)

            # visualize original, patches and skin
            if send_images_count == send_images_stride:
                send_images_count = 0
                sharedData.q_video_image.put(image)
                if Params.visualize_skin == True:
                    sharedData.q_skin_image.put(full_skin_im)
                if Params.approach == 'patches' and Params.visualize_landmarks == True:
                    annotated_image = full_skin_im.copy()
                    for idx in Params.landmarks_list:
                        cv2.circle(
                            annotated_image, (int(ldmks[idx, 1]), int(ldmks[idx, 0])), radius=0, color=Params.font_color, thickness=-1)
                        if Params.visualize_landmarks_number == True:
                            cv2.putText(annotated_image, str(idx),
                                        (int(ldmks[idx, 1]), int(ldmks[idx, 0])), cv2.FONT_HERSHEY_SIMPLEX, Params.font_size,  Params.font_color,  1)
                    if Params.visualize_patches == True:
                        if Params.patches == "squares":
                            sides = [Params.squares_dim, ] * len(magic_ldmks)
                            sides = np.array(sides)
                            annotated_image = draw_rects(
                                annotated_image, np.array(magic_ldmks[:, 1]), np.array(magic_ldmks[:, 0]), sides, sides, color)
                        elif Params.patches == "rects":
                            rects_dims = np.array(Params.rects_dims)
                            annotated_image = draw_rects(
                                annotated_image, np.array(magic_ldmks[:, 1]),
                                np.array(magic_ldmks[:, 0]), rects_dims[:, 0], rects_dims[:, 1], color)
                    # visualize patches
                    sharedData.q_patches_image.put(annotated_image)
            else:
                send_images_count += 1

            if processed_frames_count > sig_buff_dim:
                sig = sig[1:]
                if sig_buff_counter == 0:
                    sig_buff_counter = sig_stride
                    copy_sig = np.array(sig, dtype=np.float32)
                    copy_sig = np.swapaxes(copy_sig, 0, 1)
                    copy_sig = np.swapaxes(copy_sig, 1, 2)

                    ### Pre_filtering ###
                    if Params.approach == 'patches':
                        copy_sig = rgb_filter_th(copy_sig, **{'RGB_LOW_TH':  np.int32(Params.color_low_threshold),
                                                              'RGB_HIGH_TH': np.int32(Params.color_high_threshold)})
                    for filt in Params.pre_filter:
                        if filt != {}:
                            if 'fps' in filt['params'] and filt['params']['fps'] == 'adaptive' and fps is not None:
                                filt['params']['fps'] = float(fps)
                            if filt['params'] == {}:
                                copy_sig = filt['filter_func'](
                                    copy_sig)
                            else:
                                copy_sig = filt['filter_func'](
                                    copy_sig, **filt['params'])

                    ### BVP ###
                    bvp = np.zeros((0, 1), dtype=np.float32)
                    if Params.method['device_type'] == 'cpu':
                        bvp = signals_to_bvps_cpu(
                            copy_sig, Params.method['method_func'], Params.method['params'])
                    elif Params.method['device_type'] == 'torch':
                        bvp = signals_to_bvps_torch(
                            copy_sig, Params.method['method_func'], Params.method['params'])

                    ### Post_filtering ###
                    for filt in Params.pre_filter:
                        if filt != {}:
                            bvp = np.expand_dims(bvp, axis=1)
                            if 'fps' in filt['params'] and filt['params']['fps'] == 'adaptive' and fps is not None:
                                filt['params']['fps'] = float(fps)
                            if filt['params'] == {}:
                                bvp = filt['filter_func'](bvp)
                            else:
                                bvp = filt['filter_func'](
                                    bvp, **filt['params'])
                            bvp = np.squeeze(bvp, axis=1)

                    ### BPM ###
                    if BPM_obj == None:
                        BPM_obj = BPM(bvp, fps, minHz=Params.minHz,
                                        maxHz=Params.maxHz)
                    else:
                        BPM_obj.data = bvp
                    if Params.BPM_extraction_type == "welch":
                        bpm = BPM_obj.BVP_to_BPM()
                    elif Params.BPM_extraction_type == "psd_clustering":
                        bpm = BPM_obj.BVP_to_BPM_PSD_clustering()
                    if Params.approach == 'patches':  # Median of multi BPMs
                        if len(bpm.shape) > 0 and bpm.shape[0] == 0:
                            bpm = np.float32(0.0)
                        else:
                            bpm = np.float32(np.median(bpm))
                    sharedData.q_bpm.put(bpm)
                else:
                    sig_buff_counter -= 1

            end_time = time.perf_counter()*1000
            timeCount.append(end_time-start_time)
            if len(timeCount) > 100:
                timeCount = timeCount[1:]

            ### loop break ###
            if tot_frames is not None and tot_frames > 0 and processed_frames_count >= tot_frames:
                break
    if len(timeCount) > 2:
        print("Times in milliseconds of the computation of a frame:")
        print("mean:   ",statistics.mean(timeCount))
        print("median: ",statistics.median(timeCount))
        print("max:    ",max(timeCount))
        print("min:    ",min(timeCount))
    return


"""
if __name__ == "__main__":
    sd = SharedData()
    Params.videoFileName = "/home/frea/Documents/VHR/LGI_PPGI/lgi_alex/alex_resting/cv_camera_sensor_stream_handler.avi"
    Params.tot_sec = 0
    t = Thread(target=VHRroutine, args=(sd,))
    t.start()
    t.join()
"""
