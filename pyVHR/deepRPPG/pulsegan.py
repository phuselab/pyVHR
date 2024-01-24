import pickle
import os
import requests
import pyVHR
import numpy as np
import torch
import time


from .PULSEGAN.model import Generator
from .HR_CNN.utils import butter_bandpass_filter


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def PULSEGAN_bvp_pred(frames):
    print("initialize model...")
    model_path = pyVHR.__path__[0] + '/deepRPPG/PULSEGAN/pulsegan_pureubfclgi.dct'
    print(model_path)
    if not os.path.isfile(model_path):
        url = "https://github.com/Keasys/pyVHR/blob/pulsegan/resources/deepRPPG/pulsegan_pureubfclgi.dct"
        print("Downloading PulseGAN model...")
        r = requests.get(url, allow_redirects=True)
        open(model_path,"wb").write(r.content)
        
    model = Generator()
    model.load_state_dict(torch.load(model_path))
    model.to(device)
    model.eval()

    frames = torch.as_tensor(frames)
    frames = torch.utils.data.TensorDataset(frames.to(device, dtype=torch.float))
    frames_loader = torch.utils.data.DataLoader(frames,
                                                batch_size=16,
                                                shuffle=True,
                                                drop_last=True)
    
    start = time.time()
    
    outputs= []
    for i,chrom in enumerate(frames_loader):
        with torch.no_grad():
            output = model(chrom[0].unsqueeze(1)).detach().squeeze().cpu().numpy()
            outputs.append(output)
    end = time.time()

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
