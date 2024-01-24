import pyVHR
import numpy as np
import torch
import time
import torchvision.transforms as transforms

from torch.utils.data import DataLoader

from .TRANSFORMER import rPPGTransformer


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
    print(frames.shape)
    data_loader = DataLoader(frames, transforms=transforms.ToTensor())

    outputs = []

    start = time.time()
    for i, chrom_patches in enumerate(data_loader):
        output = model(chrom_patches)
        outputs.append(output)
    end = time.time()
    print("processing time: ", end - start)

    outputs = (outputs - torch.mean(outputs)) / torch.std(outputs)
    outputs = outputs.tolist()

    fs = 30
    lowcut = 0.8
    highcut = 6

    filtered_outputs = butter_bandpass_filter(outputs, lowcut, highcut, fs, order=4)
    filtered_outputs = (filtered_outputs - np.mean(filtered_outputs)) / np.std(filtered_outputs)

    return np.array(filtered_outputs)

    
    
