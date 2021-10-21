import pickle
from pyVHR.extraction.sig_processing import *
from pyVHR.BVP.BVP import *
from pyVHR.BPM.BPM import *
from pyVHR.BVP.methods import *
from pyVHR.BVP.filters import *
from pyVHR.datasets.dataset import datasetFactory
from pyVHR.utils.errors import *

######################################################################################################################
# This script computes the errors (RMSE, MAE,MAX,PCC) for each individual landmark
# and save it as a np.ndarray with shape (num_landmarks, 4) in a pickle file.

# Once you get errors for single/multiple datasets, you can use the "pyVHR_multi_heatmap.ipynb" notebook
# to compute a multi-dataset errors heatmap; this is usefull for finding the best landmarks across multiple datasets.
# We already computed some dataset errors; you can find them in "results/landmarks_errors".

# If you want to compute landmarks error for a single video please use the "pyVHR_heatmap.ipynb" notebook.
######################################################################################################################


# Save path for computed landmark errors
save_path = "/your/path/to/"

sig_extractor = SignalProcessing()
sig_extractor.display_cuda_device()
sig_extractor.choose_cuda_device(0)
sig_extractor.set_skin_extractor(SkinExtractionConvexHull('CPU'))
sig_extractor.set_total_frames(0)
SkinProcessingParams.RGB_LOW_TH = 70
SkinProcessingParams.RGB_HIGH_TH = 230
SignalProcessingParams.RGB_LOW_TH = 70
SignalProcessingParams.RGB_HIGH_TH = 230
filter_landmarks = [i for i in range(0, 468)]
sig_extractor.set_landmarks(filter_landmarks)

# Choose a dataset
dataset = datasetFactory(
    "ubfc2", videodataDIR="/var/datasets/VHR1/", BVPdataDIR="/var/datasets/VHR1/")
allvideo = dataset.videoFilenames

for v in range(len(allvideo)):
    try:
        f = open(save_path+"error_"+str(v)+".p")
        print("skipping ", v)
        continue
    except IOError:
        print(v, allvideo[v])
    wsize = 6  # seconds
    video_idx = v
    try:
        fname = dataset.getSigFilename(video_idx)
        sigGT = dataset.readSigfile(fname)
    except:
        print(v, ' skipped')
        continue
    bpmGT, timesGT = sigGT.getBPM(wsize)
    videoFileName = dataset.getVideoFilename(video_idx)
    fps = get_fps(videoFileName)

    # SIG extraction
    sig_extractor.set_square_patches_side(30.0)
    sig = sig_extractor.extract_patches(videoFileName, "squares", "mean")
    windowed_sig, timesES = sig_windowing(sig, wsize, 1, fps)

    # pre filtering
    windowed_sig = apply_filter(windowed_sig,BPfilter, params={'order':6,'minHz':0.65,'maxHz':4.0,'fps':fps})

    # rppg method
    bvps = RGB_sig_to_BVP(
        windowed_sig, fps, device_type='cuda', method=cupy_CHROM)

    # post filtering
    bvps = apply_filter(bvps, BPfilter, params={'order':6,'minHz':0.65,'maxHz':4.0,'fps':fps})

    # BVP to BPM
    bpmES = BVP_to_BPM_cuda(bvps, fps)

    ldmks_bpm = np.zeros((sig.shape[1], len(bpmES)), np.float32)
    for i in range(ldmks_bpm.shape[0]):
        for j in range(ldmks_bpm.shape[1]):
            ldmks_bpm[i, j] = bpmES[j][i]
    ldmks_err = np.zeros((sig.shape[1], 4), dtype=np.float32)
    for i in range(ldmks_err.shape[0]):
        RMSE, MAE, MAX, PCC = getErrors(np.expand_dims(
            ldmks_bpm[i, :], axis=0), bpmGT, timesES, timesGT)
        ldmks_err[i, 0] = RMSE
        ldmks_err[i, 1] = MAE
        ldmks_err[i, 2] = MAX
        if np.isnan(PCC):
            PCC = 0.0
        ldmks_err[i, 3] = PCC
    pickle.dump(ldmks_err, open(save_path+"error_"+str(video_idx)+".p", "wb"))
    print("saved ", v)
