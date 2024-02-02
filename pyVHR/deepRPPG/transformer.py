import pyVHR
import numpy as np
import os
import torch
import time
import torchvision.transforms as transforms

from torch.utils.data import DataLoader

from .TRANSFORMER.model import rPPGTransformer
from .HR_CNN.utils import butter_bandpass_filter

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def RPPG_TRANSFORMER_bvp_pred(frames):
    model_path = pyVHR.__path__[0] + "/deepRPPG/TRANSFORMER/model_vipl_next_50_1.dct"
    model_name = model_path.split('/')[-1].split('.')[0]
    if not os.path.isfile(model_path):
        url = ""
        print('Downloading rPPG Transformer model...')
        r = requests.get(url, allow_redirects=True)
        open(model_path, 'wb').write(r.content)
    model = rPPGTransformer(250, nhead=1, num_encoder_layers=12, num_decoder_layers=12).to(device) 
    model.load_state_dict(torch.load(model_path))


    model.eval()

    frames = torch.as_tensor(frames)
    frames = torch.utils.data.TensorDataset(frames.to(device, dtype=torch.float))
    frames_loader = torch.utils.data.DataLoader(frames,
                                                batch_size=16,
                                                shuffle=True,
                                                drop_last=True)

    start = time.time()

    outputs = []

    for i, chrom_patches in enumerate(frames_loader):
        output = model(chrom_patches[0]).detach().squeeze().cpu().numpy()
        outputs.append(output)
    end = time.time()
    print("processing time: ", end - start)
    
    outputs=np.array(outputs)
    
    mean = np.mean(outputs)
    outputs = (outputs - np.mean(outputs)) / np.std(outputs)

    fs = 30

    lowcut = 0.8
    highcut = 6
    
    filtered_outputs = butter_bandpass_filter(outputs, lowcut, highcut, fs, order=4)
    filtered_outputs = (filtered_outputs - np.mean(filtered_outputs)) / np.std(filtered_outputs)

    # rearrange output unbatched
    filtered_outputs = np.array(filtered_outputs)
    filtered_outputs = np.concatenate(np.concatenate(filtered_outputs, axis=0), axis=0)

    return filtered_outputs
    
    
