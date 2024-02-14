
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


movable_labels = {
    0: 0, # 'unlabeled'
    1: 0, # 'ego vehicle'
    2: 0, # 'rectification border'
    3: 0, # 'out of roi' 
    4: 0, # 'static'  
    5: 0, # 'dynamic' 
    6: 0, # 'ground'  
    7: 0, # 'road' 
    8: 0, # 'sidewalk'
    9: 0, # 'parking' 
    10: 0, # 'rail track' 
    11: 0, # 'building'
    12: 0, # 'wall' 
    13: 0, # 'fence'
    14: 0, # 'guard rail'
    15: 0, # 'bridge'
    16: 0, # 'tunnel'
    17: 0, # 'pole'
    18: 0, # 'polegroup'
    19: 0, # 'traffic light'
    20: 0, # 'traffic sign'
    21: 0, # 'vegetation'
    22: 0, # 'terrain'
    23: 0, # 'sky'
    24: 1, # 'person'
    25: 1, # 'rider'
    26: 1 , # 'car'
    27: 1, # 'truck'
    28: 1 , # 'bus'
    29: 1 , # 'caravan'
    30: 1 , # 'trailer'
    31: 1 , # 'train'
    32: 1 , # 'motorcycle'
    33: 1, # 'bicycle'
    -1: 1 , # 'license plate'
}


class CityScapesDataset(torch.utils.data.Dataset):

    def __init__(self):
        """
        """
        self.root_dir = "/home/eskotakku/Documents/Dippatyo/dataset"
        self.image_root = os.path.join(self.root_dir, "leftImg8bit_trainvaltest", "leftImg8bit", "train")
        self.truth_root = os.path.join(self.root_dir, "gtFine_trainvaltest", "gtFine", "train")

        self.items = []

        for city in os.listdir(self.image_root):
            dir = os.path.join(self.image_root, city)
            if os.path.isdir(dir):
                for file in os.listdir(dir):
                    f = os.path.join(dir, file)
                    if os.path.isfile(f):
                        color = os.path.join(self.truth_root, city, file.replace("_leftImg8bit.png", "_gtFine_labelIds.png"))
                        if not os.path.isfile(color):
                            raise Exception(color)
                        self.items.append({"image": f, "gt": color})

    def __len__(self):
        return len(self.items)


    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        image = read_image(self.items[idx]["image"], mode=ImageReadMode.RGB)
        gt = read_image(self.items[idx]["gt"], mode=ImageReadMode.UNCHANGED)

        _gt = torch.zeros(size=(2, image.shape[1], image.shape[2]), dtype=torch.bool)
        _gt[:,:,:] = torch.tensor(0, dtype=torch.bool)

        for i, val in movable_labels.items():
            Y, X = numpy.where(gt[0]==torch.tensor(i, dtype=torch.int8))
            
            if val == 1: # movable
                _gt[1, Y, X] = torch.tensor(1, dtype=torch.bool)
            else: # unmoving
                _gt[0, Y, X] = torch.tensor(1, dtype=torch.bool)
          
        # 2048x1024
        image = v2.Resize(size=256)(image)
        gt = v2.Resize(size=256, interpolation=v2.InterpolationMode.NEAREST, antialias=False)(_gt)

        # 1024 x 512
        return image.float(), gt.float()
