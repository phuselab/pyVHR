import pyVHR
import numpy as np
import torch
import torchvision.transforms as transforms
import time
from collections import OrderedDict
from torch.utils.data import DataLoader
from .HR_CNN.utils import butter_bandpass_filter
from .HR_CNN.PulseDataset import PulseDataset
from .HR_CNN.FaceHRNet09V4ELU import FaceHRNet09V4ELU


def HR_CNN_bvp_pred(frames):
    print("initialize model...")

    model_path = pyVHR.__path__[0] + '/deepRPPG/HR_CNN/hr_cnn_model.pth'
    if not os.path.isfile(model_path):
      url = "https://github.com/phuselab/pyVHR/raw/master/resources/deepRPPG/hr_cnn_model.pth"
      print('Downloading MTTS_CAN model...')
      r = requests.get(url, allow_redirects=True)
      open(model_path, 'wb').write(r.content)   

    model = FaceHRNet09V4ELU(rgb=True)

    model = torch.nn.DataParallel(model)

    model.cuda()

    ss = sum(p.numel() for p in model.parameters())
    print('num params: ', ss)

    state_dict = torch.load(model_path)

    new_state_dict = OrderedDict()
    # original saved file with DataParallel
    for k, v in state_dict.items():
        new_state_dict['module.' + k] = v

    model.load_state_dict(new_state_dict)

    pulse_test = PulseDataset(frames, transform=transforms.ToTensor())

    val_loader = DataLoader(
        pulse_test,
        batch_size=128, shuffle=False, pin_memory=True, drop_last=True)

    model.eval()

    outputs = []

    start = time.time()
    for i, net_input in enumerate(val_loader):
        net_input = net_input.cuda(non_blocking=True)

        # compute output
        with torch.no_grad():
            output = model(net_input)
            outputs.append(output.squeeze())

    end = time.time()
    print("processing time: ", end - start)

    outputs = torch.cat(outputs)

    outputs = (outputs - torch.mean(outputs)) / torch.std(outputs)

    outputs = outputs.tolist()

    fs = 30
    lowcut = 0.8
    highcut = 6

    filtered_outputs = butter_bandpass_filter(outputs, lowcut, highcut, fs, order=4)
    filtered_outputs = (filtered_outputs - np.mean(filtered_outputs)) / np.std(filtered_outputs)

    return np.array(filtered_outputs)
