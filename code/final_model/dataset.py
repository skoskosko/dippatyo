
import torch
import os
from torchvision.io import read_image
from torchvision.io import ImageReadMode
from torchvision.transforms import v2
import torchvision.transforms as transforms 
import cv2
from PIL import Image
import numpy
import sys
import tqdm
import yaml


class CityScapesDataset(torch.utils.data.Dataset):

    def __init__(self, in_memory=False):
        """
        """
        self.size = 256
        self.root_dir = os.path.abspath(os.path.join(__file__, "..", "..", ".."))
        self.l_image_root = os.path.join(self.root_dir, "dataset", "leftImg8bit_trainvaltest", "leftImg8bit", "train")
        self.r_image_root = os.path.join(self.root_dir, "dataset", "rightImg8bit_trainvaltest", "rightImg8bit", "train")
        self.truth_root = os.path.join(self.root_dir, "output")

        with open(os.path.join(self.truth_root, "clean_data.yaml")) as stream:
            data = yaml.safe_load(stream)
        
        self.items = []

        for city, images in data.items():
            for image, t in images.items():
                if t != "failure":
                    self.items.append({"city": city, "name": image, "truth_type": t})

    def __len__(self):
        return len(self.items)


    def _image(self, path):
        return v2.Resize(size=256, interpolation=v2.InterpolationMode.NEAREST, antialias=False)(read_image(path, mode=ImageReadMode.GRAY))
    
    
    
    def l_image(self, idx):
            return self._image(os.path.join(self.l_image_root, self.items[idx]["city"], self.items[idx]["name"].replace(".png", "_leftImg8bit.png")))

        
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        l_image = self.l_image(idx)
        r_image = self._image(os.path.join(self.r_image_root, self.items[idx]["city"], self.items[idx]["name"].replace(".png", "_rightImg8bit.png")))

        _image = torch.zeros(size=(3, l_image.shape[1], l_image.shape[2]), dtype=torch.uint8)
        _image[0, :, :] = l_image
        _image[1, :, :] = r_image


        _truth = self._image(os.path.join(self.truth_root, self.items[idx]["truth_type"], self.items[idx]["city"], self.items[idx]["name"]))
        _splitted_truth = torch.zeros(size=(128, l_image.shape[1], l_image.shape[2]), dtype=torch.bool)

        for i in range(128):
            Y, X = numpy.where(numpy.logical_or(_truth[0] == i*2,_truth[0] == i*2+1))
            _splitted_truth[i,Y,X] = 1

        return _image.float(), _splitted_truth.float()
