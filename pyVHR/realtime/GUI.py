from pyVHR.realtime.params import Params
from pyVHR.realtime.VHRroutine import *
import PySimpleGUI as sg
import threading
import ast
import pickle
import time

def GUI_MENU():
    col1 = [[sg.Text("Set Parameters:")],
            [sg.Text("Video file path, or device number:")],
            [sg.In("", key="-VideoFileName-"),
             sg.FileBrowse(target=("-VideoFileName-"))],
            [sg.Text("Window Size:"),
             sg.In(str(Params.winSize), key="-winSize-", size=(5, 1))],
            [sg.Text("Stride:"),
             sg.In(str(Params.stride), key="-stride-", size=(5, 1))],
            [sg.Text("FPS fixed:"),
             sg.In("None", key="-fpsfixed-", size=(5, 1))],
            [sg.Text("            ")],
            [sg.Text("Skin Extractor:"),
             sg.Radio("Convex Hull", "RADIO2",
                      default=True, key="-skinConvex-"),
             sg.Radio("FaceParsing", "RADIO2", default=False, key="-skinFaceParsing-")],
            [sg.Text("ROI:"),
             sg.Radio("Holistic", "RADIO3",
                      default=False, key="-holistic-"),
             sg.Radio("Patches", "RADIO3", default=True, key="-patches-")],
            [sg.Text("Method for Signal computation:"),
             sg.Radio("mean", "RADIO0",
                      default=True, key="-mean-"),
             sg.Radio("median", "RADIO0", default=False, key="-median-")],
            [sg.Text("            ")],
            [sg.Text("Patches type:"),
             sg.Radio("Squares", "RADIO4",
                      default=True, key="-squares-"),
             sg.Radio("Rectangles", "RADIO4", default=False, key="-rects-")],
            [sg.Text("Landmarks list (ex: [1,2,3]):")],
            [sg.In("Default", key="-ldmkslist-")],
            [sg.Text("Squares dimension:"),
             sg.In(str(Params.squares_dim), key="-squares_dim-", size=(5, 1))],
            [sg.Text("Rects dimension (ex for two landmarks: [[20,15],[10,10]]):")],
            [sg.In("[[],]", key="-rectsdim-")],
            [sg.Text("            ")],
            [sg.Button('Apply Parameters', key='-APPLY-')],
            [sg.Button('START', key='-START-', visible=False)]
            ]
    col2 = [[sg.Text("            ")],
            [sg.Text("Skin RGB color low threshold (ex: input V -> RGB(V,V,V)):       "),
             sg.In(str(Params.skin_color_low_threshold), key="-skin_color_low_threshold-", size=(5, 1))],
            [sg.Text("Skin RGB color high threshold (ex: input V -> RGB(V,V,V)):      "),
             sg.In(str(Params.skin_color_high_threshold), key="-skin_color_high_threshold-", size=(5, 1))],
            [sg.Text("ROI RGB color low threshold (ex: input V -> RGB(V,V,V)):        "),
             sg.In(str(Params.sig_color_low_threshold), key="-sig_color_low_threshold-", size=(5, 1))],
            [sg.Text("ROI RGB color high threshold (ex: input V -> RGB(V,V,V)):       "),
             sg.In(str(Params.sig_color_high_threshold), key="-sig_color_high_threshold-", size=(5, 1))],
            [sg.Text("            ")],
            [sg.Text("Signal RGB color low threshold (ex: input V -> RGB(V,V,V)):     "),
             sg.In(str(Params.color_low_threshold), key="-color_low_threshold-", size=(5, 1))],
            [sg.Text("Signal RGB color high threshold (ex: input V -> RGB(V,V,V)):    "),
             sg.In(str(Params.color_high_threshold), key="-color_high_threshold-", size=(5, 1))],
            [sg.Text("            ")],
            [sg.Text("Visualize Skin:"),
             sg.Radio("True", "RADIO5",
                      default=True, key="-visualizeskintrue-"),
             sg.Radio("False", "RADIO5", default=False, key="-visualizeskinfalse-")],
            [sg.Text("Visualize Landmarks:"),
             sg.Radio("True", "RADIO6",
                      default=True, key="-visualizeldmkstrue-"),
             sg.Radio("False", "RADIO6", default=False, key="-visualizeldmksfalse-")],
            [sg.Text("\t - Visualize Patches:"),
             sg.Radio("True", "RADIO7",
                      default=True, key="-visualizepatchestrue-"),
             sg.Radio("False", "RADIO7", default=False, key="-visualizepatchesfalse-")],
            [sg.Text("\t - Visualize Landmarks number:"),
             sg.Radio("True", "RADIO8",
                      default=True, key="-visualizeldmksnumtrue-"),
             sg.Radio("False", "RADIO8", default=False, key="-visualizeldmksnumfalse-")],
            [sg.Text("            \n Font Color (R,G,B,A)")],
            [sg.In(str(Params.font_color), key='-FontColor-', size=(25, 1))],
            [sg.Text("Font Size: "), sg.In(
                str(Params.font_size), key='-FontSize-', size=(4, 1))],
            [sg.Text("For Editing PRE-POST filters, BVP method and BPM params edit:\n Params.pre_filter, Params.method, Params.post_filter,\n Params.minHz, Params.maxHz")]
            ]
    layout = [[sg.Column(col1, element_justification='l'),
              sg.Column(col2, element_justification='l')]]
    window = sg.Window('pyVHR', layout, finalize=True, location=(0, 0))

    bpm_plot = []
    bpm_save = []
    bpm_plot_max = 60
    image = None
    original_image = None
    skin_image = None
    patches_image = None
    vhr_t = None
    sd = SharedData()

    def create_run_window():
        fps = Params.fps_fixed if Params.fps_fixed is not None else get_fps(
            Params.videoFileName)
        visualize_list = [[sg.Radio(
            'Original Video',  "RADIO11", default=True, key="-IN_Visualize_original-")], ]
        if Params.visualize_skin:
            visualize_list.append(
                [sg.Radio('Skin', "RADIO11", default=False, key="-IN_Visualize_skin-")])
        if Params.approach == 'patches':
            visualize_list.append(
                [sg.Radio('Patches', "RADIO11", default=False, key="-IN_Visualize_patches-")])

        run_col1 = [[sg.Text(Params.videoFileName)],
                    [sg.Text(
                        "Video FPS: "+str(fps)),
                    sg.Text(
                        "Video resolution:             ", key="-res-")],
                    [sg.Text("Visualize: ")],
                    *visualize_list,
                    [sg.Text('BPMs')],
                    [sg.Image(filename='', key='-BPM_PLOT-')],
                    [sg.Button('Stop')],
                    [sg.Button('Save BPMs')]]
        run_col2 = [[sg.Text('Visualize')],
                    [sg.Image(filename='', key='-VIDEO-')]]
        run_layout = [[sg.Column(run_col1, element_justification='l'),
                       sg.Column(run_col2, element_justification='c')]]
        wr = sg.Window('pyVHR - RUN', run_layout,
                       finalize=True, location=(0, 0))
        # blank plots for compiling kernels
        imgbytes = cv2.imencode('.png', cv2.cvtColor(
            np.zeros((480, 640, 3), dtype=np.uint8), cv2.COLOR_BGR2RGB))[1].tobytes()
        wr['-VIDEO-'].update(data=imgbytes)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=np.arange(0, 10, 1), y=np.zeros(9),
                                 mode='lines+markers',
                                 name='lines+markers'))
        fig.update_layout(autosize=False, width=400, height=300, margin=dict(
            l=1, r=1, b=1, t=1, pad=1), paper_bgcolor="LightSteelBlue",)
        img_bytes = fig.to_image(format="png")
        wr['-BPM_PLOT-'].update(data=img_bytes)
        return wr

    run_window = None
    while True:
        win, event, values = sg.read_all_windows(timeout=16)
        if event in (None, 'Exit', 'Stop', sg.WIN_CLOSED) and run_window is not None:
            sd.q_stop_cap.put(0)  # stop cap, it will stop VHR model thread
            sd.q_stop.put(0)  # stop VHR model thread if cap can't
            vhr_t.join()
            vhr_t = None
            sd = None
            sd = SharedData()
            run_window.close()
            run_window = None
        elif event in (None, 'Exit', sg.WIN_CLOSED) and run_window is None:
            window.close()
            break
        if run_window is not None:
            run_win_event, run_win_values = run_window.read(timeout=16)
            # Video plot
            if not sd.q_video_image.empty():
                original_image = sd.q_video_image.get()
            if not sd.q_skin_image.empty():
                skin_image = sd.q_skin_image.get()
            if not sd.q_patches_image.empty():
                patches_image = sd.q_patches_image.get()
            if original_image is not None and bool(run_win_values["-IN_Visualize_original-"]):
                image = original_image
            elif Params.visualize_skin and skin_image is not None and bool(run_win_values["-IN_Visualize_skin-"]):
                image = skin_image
            elif Params.visualize_landmarks and patches_image is not None and bool(run_win_values["-IN_Visualize_patches-"]):
                image = patches_image
            if image is not None:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                imgbytes = cv2.imencode('.png', image)[1].tobytes()
                run_window['-VIDEO-'].update(data=imgbytes)
                run_window['-res-'].update("Video resolution: " +
                                           str(image.shape[1])+" x "+str(image.shape[0]))
            # BPM plot
            if not sd.q_bpm.empty():
                bpm = sd.q_bpm.get()
                bpm_plot.append(bpm)
                bpm_save.append(bpm)
                if len(bpm_plot) >= bpm_plot_max:
                    bpm_plot = bpm_plot[1:]
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=np.arange(0, len(bpm_plot), Params.stride), y=np.array(bpm_plot),
                                         mode='lines+markers',
                                         name='lines+markers'))
                fig.update_layout(
                    autosize=False,
                    width=600,
                    height=400,
                    margin=dict(l=1, r=1, b=1, t=1, pad=1),
                    paper_bgcolor="LightSteelBlue",
                )
                img_bytes = fig.to_image(format="png")
                run_window['-BPM_PLOT-'].update(data=img_bytes)
            if run_win_event == 'Save BPMs':
                try:
                    path = Params.out_path if Params.out_path is not None else os.getcwd() + \
                        os.path.sep
                    with open(str(path) + str(time.time())+'_BPMs', 'wb') as f:
                        pickle.dump(bpm_save, f)
                    print("BPMs saved!")
                except:
                    print("[ERROR] wrong path for saving BPMs!")
        if event == '-APPLY-':
            if values['-VideoFileName-'] == '':
                Params.videoFileName = 0
                window['-VideoFileName-'].update('0')
            else:
                Params.videoFileName = values['-VideoFileName-']
            if values['-VideoFileName-'].isnumeric():
                Params.videoFileName = int(values['-VideoFileName-'])
                Params.fake_delay = False
            else:
                Params.fake_delay = True
            if values['-winSize-'].isnumeric():
                Params.winSize = int(
                    values['-winSize-']) if int(values['-winSize-']) > 1 else 3
            window['-winSize-'].update(str(Params.winSize))
            if values['-stride-'].isnumeric():
                Params.stride = int(
                    values['-stride-']) if int(values['-stride-']) > 0 else 1
            window['-stride-'].update(str(Params.stride))
            Params.cuda = False
            if values['-fpsfixed-'].isnumeric():
                Params.fps_fixed = int(
                    values['-fpsfixed-']) if int(values['-fpsfixed-']) > 0 else None
            window['-fpsfixed-'].update(str(Params.fps_fixed))
            Params.skin_extractor = 'convexhull' if bool(
                values['-skinConvex-']) else 'faceparsing'
            Params.approach = 'holistic' if bool(
                values['-holistic-']) else 'patches'
            Params.type = 'mean' if bool(
                values['-mean-']) else 'median'
            Params.patches = 'squares' if bool(
                values['-squares-']) else 'rects'
            try:
                ldlist = ast.literal_eval(values['-ldmkslist-'])
                Params.landmarks_list = ldlist if len(
                    ldlist) > 0 else Params.landmarks_list
            except:
                window['-ldmkslist-'].update(str(Params.landmarks_list))
            if values['-squares_dim-'].isnumeric():
                Params.squares_dim = float(
                    values['-squares_dim-']) if float(values['-squares_dim-']) > 0 else None
            else:
                window['-squares_dim-'].update(str(Params.squares_dim))
            try:
                Params.rects_dims = ast.literal_eval(values['-rectsdim-'])
                # Default if len is not the same
                if len(Params.rects_dims) != len(Params.landmarks_list):
                    new_rect_dim = []
                    for i in range(len(Params.landmarks_list)):
                        new_rect_dim.append(
                            [Params.squares_dim, Params.squares_dim])
                    Params.rects_dims = new_rect_dim
                    window['-rectsdim-'].update(str(new_rect_dim))
            except:
                # Default if parameter is wrong
                new_rect_dim = []
                for i in range(len(Params.landmarks_list)):
                    new_rect_dim.append(
                        [Params.squares_dim, Params.squares_dim])
                Params.rects_dims = new_rect_dim
                window['-rectsdim-'].update(str(new_rect_dim))

            if values['-skin_color_low_threshold-'].isnumeric():
                Params.skin_color_low_threshold = int(
                    values['-skin_color_low_threshold-']) if int(values['-skin_color_low_threshold-']) >= 0 and int(values['-skin_color_low_threshold-']) <= 255 else 2
            window['-skin_color_low_threshold-'].update(
                str(Params.skin_color_low_threshold))

            if values['-skin_color_high_threshold-'].isnumeric():
                Params.skin_color_high_threshold = int(
                    values['-skin_color_high_threshold-']) if int(values['-skin_color_high_threshold-']) >= 0 and int(values['-skin_color_high_threshold-']) <= 255 else 254
            window['-skin_color_high_threshold-'].update(
                str(Params.skin_color_high_threshold))

            if values['-sig_color_low_threshold-'].isnumeric():
                Params.sig_color_low_threshold = int(
                    values['-sig_color_low_threshold-']) if int(values['-sig_color_low_threshold-']) >= 0 and int(values['-sig_color_low_threshold-']) <= 255 else 2
            window['-sig_color_low_threshold-'].update(
                str(Params.sig_color_low_threshold))

            if values['-sig_color_high_threshold-'].isnumeric():
                Params.sig_color_high_threshold = int(
                    values['-sig_color_high_threshold-']) if int(values['-sig_color_high_threshold-']) >= 0 and int(values['-sig_color_high_threshold-']) <= 255 else 254
            window['-sig_color_high_threshold-'].update(
                str(Params.sig_color_high_threshold))

            if values['-color_low_threshold-'].isnumeric():
                Params.color_low_threshold = int(
                    values['-color_low_threshold-']) if int(values['-color_low_threshold-']) >= 0 and int(values['-color_low_threshold-']) <= 255 else -1
            window['-color_low_threshold-'].update(
                str(Params.color_low_threshold))

            if values['-color_high_threshold-'].isnumeric():
                Params.color_high_threshold = int(
                    values['-color_high_threshold-']) if int(values['-color_high_threshold-']) >= 0 and int(values['-color_high_threshold-']) <= 255 else 255
            window['-color_high_threshold-'].update(
                str(Params.color_high_threshold))

            Params.visualize_skin = True if bool(
                values['-visualizeskintrue-']) else False
            Params.visualize_landmarks = True if bool(
                values['-visualizeldmkstrue-']) else False
            Params.visualize_patches = True if bool(
                values['-visualizepatchestrue-']) else False
            Params.visualize_landmarks_number = True if bool(
                values['-visualizeldmksnumtrue-']) else False

            try:
                colorfont = ast.literal_eval(values['-FontColor-'])
                if len(colorfont) == 4:
                    correct = True
                    for e in colorfont:
                        if not(e >= 0 and e <= 255):
                            correct = False
                    if correct:
                        Params.font_color = colorfont
            except:
                pass
            window['-FontColor-'].update(str(Params.font_color))

            if values['-FontSize-'].isnumeric():
                Params.font_size = float(
                    values['-FontSize-']) if float(values['-FontSize-']) > 0.0 else 0.3
            window['-FontSize-'].update(str(Params.font_size))
            window['-START-'].update(visible=True)
        if event == '-START-' and not run_window:
            bpm_plot = []
            bpm_save = []
            image = None
            sd = SharedData()
            run_window = create_run_window()
            if vhr_t == None:
                vhr_t = threading.Thread(target=VHRroutine, args=(sd,))
                vhr_t.daemon = False
                vhr_t.start()
    window.close()


if __name__ == '__main__':
    # Theese parameters can be edited only here.

    # Pre and Post filters are list of dictionary with the following structure:
    # {'filter_func': method_name, 'params': {}}

    # A rPPG method is a dictionary with the following structure:
    # {'method_func': method_name, 'device_type': 'device_name', 'params': {}}
    # device_type can be: 'cpu', 'torch'.

    Params.pre_filter = [{'filter_func': BPfilter, 'params': {
        'minHz': 0.7, 'maxHz': 3.0, 'fps': 'adaptive', 'order': 6}}]
        
    Params.method = {'method_func': cpu_CHROM,
                     'device_type': 'cpu', 'params': {}}

    Params.post_filter = [{'filter_func': BPfilter, 'params': {
        'minHz': 0.7, 'maxHz': 3.0, 'fps': 'adaptive', 'order': 6}}]

    # BPM extraction type
    # 'psd_clustering' or welch
    Params.BPM_extraction_type = 'welch'

    # Saving path for BPMs
    #Params.out_path = "/home/user/Downloads/"
    Params.out_path = None  # Working directory will be used

    # Downscale input video to width 640 (preserve aspect-ratio)
    Params.resize = False

    GUI_MENU()
