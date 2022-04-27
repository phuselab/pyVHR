import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image


class PulseDataset(Dataset):
    """
    Pulse dataset.
    Frames in shape [c x w x h].
    """

    def __init__(self, frames, img_w=128, img_h=192, transform=None):
        """
        Initialize dataset
        :param frames: video frames
        :param img_h: height of frame
        :param img_w: width of frame
        :param transform: transforms to apply to data
        """

        self.frames_list = frames
        self.img_w = img_w
        self.img_h = img_h
        self.transform = transform
        print('Found', self.__len__(), "frames")

    def __len__(self):
        return self.frames_list.__len__()

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        mm = []
        fr = self.frames_list[idx]
        image = Image.fromarray(fr)
        image = image.resize((self.img_w, self.img_h))

        _, b, _ = image.split()
        mean_img = np.mean(b)
        mm.append(mean_img)
        if self.transform:
            image = self.transform(image)

        image = torch.as_tensor(image)
        image = (image - torch.mean(image)) / torch.std(image) * 255

        return image
