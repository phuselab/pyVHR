from __future__ import print_function
import matplotlib.pyplot as plt
import PIL
from ipywidgets import interact, interactive, fixed, interact_manual
import ipywidgets as widgets
import plotly.graph_objects as go
import plotly.express as px
from PIL import Image
from IPython.display import display, clear_output
import cv2
import mediapipe as mp
import numpy as np
import pyVHR
from pyVHR.extraction.sig_processing import extract_frames_yield
from scipy.signal import welch
import random



"""
This module defines classes or methods used for plotting outputs.
"""


class VisualizeParams:
    """
    This class contains usefull parameters used by this module.

    The "renderer" variable is used for rendering plots on Jupyter notebook ('notebook')
    or Colab notebook ('colab').
    """
    renderer = 'colab'  # or 'notebook'

def interactive_image_plot(images_list, scaling=1):
    """
    This method create an interactive plot of a list of images. This method must be called
    inside a Jupyter notebook or a Colab notebook.

    Args:
        images_list (list of ndarray): list of images as ndarray with shape [rows, columns, rgb_channels].
        scaling (float): scale factor useful for enlarging or decreasing the resolution of the image.
    
    """
    if images_list is None or len(images_list) == 0:
        return

    def f(x):
        PIL_image = PIL.Image.fromarray(np.uint8(images_list[x]))
        width, height = PIL_image.size
        PIL_image = PIL_image.resize(
            (int(width*scaling), int(height*scaling)), PIL.Image.NEAREST)
        display(PIL_image)
    interact(f, x=widgets.IntSlider(
        min=0, max=len(images_list)-1, step=1, value=0))


def display_video(video_file_name, scaling=1):
    """
    This method create an interactive plot for visualizing the frames of a video. This method must be called
    inside a Jupyter notebook or a Colab notebook.

    Args:
        video_file_name (str): video file name or path.
        scaling (float): scale factor useful for enlarging or decreasing the resolution of the image.
    
    """
    original_frames = [cv2.cvtColor(_, cv2.COLOR_BGR2RGB)
                       for _ in extract_frames_yield(video_file_name)]
    interactive_image_plot(original_frames, scaling)


def visualize_windowed_sig(windowed_sig, window):
    """
    This method create a plotly plot for visualizing a window of a windowed signal. This method must be called
    inside a Jupyter notebook or a Colab notebook.

    Args:
        windowed_sig (list of float32 ndarray): windowed signal as a list of length num_windows of float32 ndarray with shape [num_estimators, rgb_channels, window_frames].
        window (int): the index of the window to plot.
    
    """
    fig = go.Figure()
    i = 1
    for e in windowed_sig[window]:
        name = "sig_" + str(i) + "_r"
        r_color = random.randint(1, 255)
        g_color = random.randint(1, 255)
        b_color = random.randint(1, 255)
        fig.add_trace(go.Scatter(x=np.arange(e.shape[-1]), y=e[0, :],
                                 mode='lines', marker_color='rgba('+str(r_color)+', '+str(g_color)+', '+str(b_color)+', 1.0)', name=name))
        name = "sig_" + str(i) + "_g"
        fig.add_trace(go.Scatter(x=np.arange(e.shape[-1]), y=e[1, :],
                                 mode='lines', marker_color='rgba('+str(r_color)+', '+str(g_color)+', '+str(b_color)+', 1.0)', name=name))
        name = "sig_" + str(i) + "_b"
        fig.add_trace(go.Scatter(x=np.arange(e.shape[-1]), y=e[2, :],
                                 mode='lines', marker_color='rgba('+str(r_color)+', '+str(g_color)+', '+str(b_color)+', 1.0)', name=name))
        i += 1
    fig.update_layout(title="WIN #" + str(window))
    fig.show(renderer=VisualizeParams.renderer)


def visualize_BVPs(BVPs, window):
    """
    This method create a plotly plot for visualizing a window of a windowed BVP signal. This method must be called
    inside a Jupyter notebook or a Colab notebook.

    Args:
        BVPs (list of float32 ndarray): windowed BPM signal as a list of length num_windows of float32 ndarray with shape [num_estimators, window_frames].
        window (int): the index of the window to plot.
    
    """
    fig = go.Figure()
    i = 1
    bvp = BVPs[window]
    for e in bvp:
        name = "BVP_" + str(i)
        fig.add_trace(go.Scatter(x=np.arange(bvp.shape[1]), y=e[:],
                                 mode='lines', name=name))
        i += 1
    fig.update_layout(title="BVP #" + str(window))
    fig.show(renderer=VisualizeParams.renderer)


def visualize_multi_est_BPM_vs_BPMs_list(multi_est_BPM, BPMs_list):
    """
    This method create a plotly plot for visualizing a multi-estimator BPM signal and a list of BPM signals. 
    This is usefull when comparing Patches BPMs vs Holistic and Ground Truth BPMs. This method must be called
    inside a Jupyter notebook or a Colab notebook.

    Args:
        multi_est_BPM (list): multi-estimator BPM signal is a list that contains two elements [mul-est-BPM, times]; the first contains BPMs as a list of 
            length num_windows of float32 ndarray with shape [num_estimators, ], the second is a float32 1D ndarray that contains the time in seconds of each BPM.
        BPMs_list (list): The BPM signals is a 2D list structured as [[BPM_list, times, name_tag], ...]. The first element is a float32 ndarray that
            contains the BPM signal, the second element is a float32 1D ndarray that contains the time in seconds of each BPM, the third element is a string
            that is used in the plot's legend.    
    """
    fig = go.Figure()
    for idx, _ in enumerate(BPMs_list):
        name = str(BPMs_list[idx][2])
        fig.add_trace(go.Scatter(x=BPMs_list[idx][1], y=BPMs_list[idx][0], mode='lines', name=name))
    for w, _ in enumerate(multi_est_BPM[0]):
        name = "BPMs_" + str(w+1)
        data = multi_est_BPM[0][w]
        if data.shape == ():
            t = [multi_est_BPM[1][w], ]
            data = [multi_est_BPM[0][w], ]
        else:
            t = multi_est_BPM[1][w] * np.ones(data.shape[0])
        fig.add_trace(go.Scatter(x=t, y=data,mode='markers', marker=dict(size=2), name=name))
    fig.update_layout(title="BPMs estimators vs BPMs list", xaxis_title="Time", yaxis_title="BPM")
    fig.show(renderer=VisualizeParams.renderer)

def visualize_BPMs(BPMs_list):
    """
    This method create a plotly plot for visualizing a list of BPM signals. This method must be called
    inside a Jupyter notebook or a Colab notebook.

    Args:
        BPMs_list (list): The BPM signals is a 2D list structured as [[BPM_list, times, name_tag], ...]. The first element is a float32 ndarray that
            contains the BPM signal, the second element is a float32 1D ndarray that contains the time in seconds of each BPM, the third element is a string
            that is used in the plot's legend.    
    """
    fig = go.Figure()
    i = 1
    for e in BPMs_list:
        name = str(e[2])
        fig.add_trace(go.Scatter(x=e[1], y=e[0],
                                 mode='lines+markers', name=name))
        i += 1
    fig.update_layout(title="BPMs")
    fig.show(renderer=VisualizeParams.renderer)


def visualize_BVPs_PSD(BVPs, window, fps, minHz=0.65, maxHz=4.0):
    """
    This method create a plotly plot for visualizing the Power Spectral Density of a window of a windowed BVP signal. This method must be called
    inside a Jupyter notebook or a Colab notebook.

    Args:
        BVPs (list of float32 ndarray): windowed BPM signal as a list of length num_windows of float32 ndarray with shape [num_estimators, window_frames].
        window (int): the index of the window to plot.
        fps (float): frames per seconds.
        minHz (float): frequency in Hz used to isolate a specific subband [minHz, maxHz] (esclusive).
        maxHz (float): frequency in Hz used to isolate a specific subband [minHz, maxHz] (esclusive).
    
    """
    data = BVPs[window]
    _, n = data.shape
    if data.shape[0] == 0:
        return np.float32(0.0)
    if n < 256:
        seglength = n
        overlap = int(0.8*n)  # fixed overlapping
    else:
        seglength = 256
        overlap = 200
    # -- periodogram by Welch
    F, P = welch(data, nperseg=seglength,
                 noverlap=overlap, fs=fps, nfft=2048)
    F = F.astype(np.float32)
    P = P.astype(np.float32)
    # -- freq subband (0.65 Hz - 4.0 Hz)
    band = np.argwhere((F > minHz) & (F < maxHz)).flatten()
    Pfreqs = 60*F[band]
    Power = P[:, band]
    # -- BPM estimate by PSD
    Pmax = np.argmax(Power, axis=1)  # power max

    # plot
    fig = go.Figure()
    for idx in range(P.shape[0]):
        fig.add_trace(go.Scatter(
            x=F*60, y=P[idx], name="PSD_"+str(idx)+" no band"))
        fig.add_trace(go.Scatter(
            x=Pfreqs, y=Power[idx], name="PSD_"+str(idx)+" band"))
    fig.update_layout(title="PSD #" + str(window), xaxis_title='Beats per minute [BPM]')
    fig.show(renderer=VisualizeParams.renderer)


def visualize_landmarks_list(image_file_name=None, landmarks_list=None):
    """
    This method create a plotly plot for visualizing a list of facial landmarks on a given image. This is useful
    for studying and analyzing the available facial points of MediaPipe (https://google.github.io/mediapipe/solutions/face_mesh.html).
    This method must be called inside a Jupyter notebook or a Colab notebook.

    Args:
        image_file_name (str): image file name or path (preferred png or jpg).
        landmarks_list (list): list of positive integers between 0 and 467 that identify patches centers (landmarks).
    
    """
    PRESENCE_THRESHOLD = 0.5
    VISIBILITY_THRESHOLD = 0.5
    if image_file_name is None:
        image_file_name = pyVHR.__path__[0] + '/resources/img/face.png' 
    imag = cv2.imread(image_file_name, cv2.COLOR_RGB2BGR)
    imag = cv2.cvtColor(imag, cv2.COLOR_BGR2RGB)
    mp_drawing = mp.solutions.drawing_utils
    mp_face_mesh = mp.solutions.face_mesh
    with mp_face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            min_detection_confidence=0.5) as face_mesh:
        image = cv2.imread(image_file_name)
        results = face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        if not results.multi_face_landmarks:
            return
        width = image.shape[1]
        height = image.shape[0]
        face_landmarks = results.multi_face_landmarks[0]
        ldmks = np.zeros((468, 3), dtype=np.float32)
        for idx, landmark in enumerate(face_landmarks.landmark):
            if ((landmark.HasField('visibility') and landmark.visibility < VISIBILITY_THRESHOLD)
                    or (landmark.HasField('presence') and landmark.presence < PRESENCE_THRESHOLD)):
                ldmks[idx, 0] = -1.0
                ldmks[idx, 1] = -1.0
                ldmks[idx, 2] = -1.0
            else:
                coords = mp_drawing._normalized_to_pixel_coordinates(
                    landmark.x, landmark.y, width, height)
                if coords:
                    ldmks[idx, 0] = coords[0]
                    ldmks[idx, 1] = coords[1]
                    ldmks[idx, 2] = idx
                else:
                    ldmks[idx, 0] = -1.0
                    ldmks[idx, 1] = -1.0
                    ldmks[idx, 2] = -1.0
    
    filtered_ldmks = []
    if landmarks_list is not None:
        for idx in landmarks_list:
            filtered_ldmks.append(ldmks[idx])
        filtered_ldmks = np.array(filtered_ldmks, dtype=np.float32)
    else:
        filtered_ldmks = ldmks

    fig = px.imshow(imag)
    for l in filtered_ldmks:
        name = 'ldmk_' + str(int(l[2]))
        fig.add_trace(go.Scatter(x=(l[0],), y=(l[1],), name=name, mode='markers', 
                                marker=dict(color='blue', size=3)))
    fig.update_xaxes(range=[0,imag.shape[1]])
    fig.update_yaxes(range=[imag.shape[0],0])
    fig.update_layout(paper_bgcolor='#eee') 
    fig.show(renderer=VisualizeParams.renderer)


from pyVHR.BPM.utils import Model, gaussian, Welch, Welch_cuda, pairwise_distances, circle_clustering, optimize_partition, gaussian_fit
def visualize_BVPs_PSD_clutering(GT_BPM, GT_times , BVPs, times, fps, minHz=0.65, maxHz=4.0, out_fact=1):
    """
    TODO: documentare
    This method create a plotly plot for visualizing the Power Spectral Density of a window of a windowed BVP signal. This method must be called
    inside a Jupyter notebook or a Colab notebook.

    Args:
        BVPs (list of float32 ndarray): windowed BPM signal as a list of length num_windows of float32 ndarray with shape [num_estimators, window_frames].
        window (int): the index of the window to plot.
        fps (float): frames per seconds.
        minHz (float): frequency in Hz used to isolate a specific subband [minHz, maxHz] (esclusive).
        maxHz (float): frequency in Hz used to isolate a specific subband [minHz, maxHz] (esclusive).
    
    """  
    gmodel = Model(gaussian, independent_vars=['x', 'mu', 'a'])
    for i,X in enumerate(BVPs):
        if X.shape[0] == 0:
            continue
        F, PSD = Welch(X, fps, minHz, maxHz)
        W = pairwise_distances(PSD, PSD, metric='cosine')
        #W = np.corrcoef(PSD)-np.eye(W.shape[0])
        #W = dtw.distance_matrix_fast(PSD.astype(np.double))
        theta = circle_clustering(W, eps=0.01)

        # bi-partition, sum and normalization
        P, Q, Z, med_elem_P, med_elem_Q = optimize_partition(theta, out_fact=out_fact)

        P0 = np.sum(PSD[P,:], axis=0)
        max = np.max(P0, axis=0)
        max = np.expand_dims(max, axis=0)
        P0 = np.squeeze(np.divide(P0, max))

        P1 = np.sum(PSD[Q,:], axis=0)
        max = np.max(P1, axis=0)
        max = np.expand_dims(max, axis=0)
        P1 = np.squeeze(np.divide(P1, max))
        
        # peaks
        peak0_idx = np.argmax(P0) 
        P0_max = P0[peak0_idx]
        F0 = F[peak0_idx]

        peak1_idx = np.argmax(P1) 
        P1_max = P1[peak1_idx]
        F1 = F[peak1_idx]
        
        # Gaussian fitting
        result0, G1, sigma0 = gaussian_fit(gmodel, P0, F, F0, 1)  # Gaussian fit 
        result1, G2, sigma1 = gaussian_fit(gmodel, P1, F, F1, 1)  # Gaussian fit 
        chisqr0 = result0.chisqr
        chisqr1 = result1.chisqr

        print('** Processing window n. ', i)
        t = np.argmin(np.abs(times[i]-GT_times))
        GT = GT_BPM[t]
        print('GT = ', GT, '   freq0 max = ', F0, '   freq1 max = ', F1)
        # TODO: rifare con plotly
        plt.figure(figsize=(14, 4))
        plt.subplot(121)
        plt.plot(F, PSD[P,:].T, alpha=0.2)
        _, top = plt.ylim() 
        plt.plot(F, 0.5*top*P0, color='blue')
        
        plt.subplot(122)
        plt.ylim(0,1.1) 
        plt.plot(F, result0.best_fit, linestyle='-', color='magenta')
        plt.fill_between(F, result0.best_fit, alpha=0.1)
        plt.plot(F, P0, color='blue')
        plt.vlines(F0, ymin=0, ymax=1, linestyle='-.', color='blue')
        plt.vlines(GT,  ymin=0, ymax=1.1, linestyle='-.', color='black')
        plt.title('err = '+ str(np.round(np.abs(GT-F0),2)) +' -- sigma = '+ str(np.round(sigma0,2)) + ' -- chi sqr = ' + str(str(np.round(chisqr0,2))))
        plt.show()

        # second figure
        plt.figure(figsize=(14, 4))
        plt.subplot(121)
        plt.plot(F, PSD[Q,:].T, alpha=0.2)
        _, top = plt.ylim() 
        plt.plot(F, 0.5*top*P1, color='red')
        plt.subplot(122)
        plt.ylim(0,1.1) 
        plt.plot(F, result1.best_fit, linestyle='-', color='magenta')
        plt.fill_between(F, result1.best_fit, alpha=0.1)
        plt.plot(F, P1, color='red')
        plt.vlines(F1, ymin=0, ymax=1, linestyle='-.', color='red')
        plt.vlines(GT,  ymin=0, ymax=1.1, linestyle='-.', color='black')
        plt.title('err = '+ str(np.round(np.abs(GT-F1),2)) +' -- sigma = '+ str(np.round(sigma1,2)) + ' -- chi sqr = ' + str(str(np.round(chisqr1,2))))
        plt.show()

        # centers
        C1 = [np.cos(med_elem_P), np.sin(med_elem_P)]
        C2 = [np.cos(med_elem_Q), np.sin(med_elem_Q)]

        #hist, bins = histogram(theta, nbins=256)
        labels = np.zeros_like(theta)
        labels[Q] = 1
        labels[Z] = 2
        plot_circle(theta, l=labels, C1=C1, C2=C2)


  
def plot_circle(theta, l=None, C1=None, C2=None, radius=500):
    """
    TODO: documentare
    Produce a plot with the locations of all poles and zeros
    """

    x = np.cos(theta)
    y = np.sin(theta)

    fig = go.Figure()
    fig.add_shape(type="circle", xref="x", yref="y", x0=-1, y0=-1, x1=1, y1=1, line=dict(color="black", width=1))
    
    if l is None:
        fig.add_trace(go.Scatter(x=x, y=y,
            mode='markers',
            marker_symbol='circle',
            marker_size=10))
    else:
        ul = np.unique(l)
        cols = ['blue', 'red', 'gray']
        for c,u in zip(cols,ul):
            idx = np.where(u == l)
            fig.add_trace(go.Scatter(x=x[idx], y=y[idx],
                mode='markers',
                marker_symbol='circle',
                marker_color=c, 
                # marker_line_color=cols[c],
                marker_line_width=0, 
                marker_size=10))
        
    # separator plane
    fig.add_trace(go.Scatter(x=[0.0, C1[0]], y=[0.0, C1[1]],
        mode='lines+markers',
        marker_symbol='circle',
        marker_color='blue', 
        marker_line_color='blue',
        marker_line_width=2, 
        marker_size=2))
    fig.add_trace(go.Scatter(x=[0.0, C2[0]], y=[0.0, C2[1]],
        mode='lines+markers',
        marker_symbol='circle',
        marker_color='red', 
        marker_line_color='red',
        marker_line_width=2, 
        marker_size=2))
    
    M = 1.05
    fig.update_xaxes(title='', range=[-M, M])
    fig.update_yaxes(title='', range=[-M, M])
    fig.update_layout(title='clusters', width=radius, height=radius)
    fig.show(renderer=VisualizeParams.renderer)
