import os
from typing import List
from torchvision.io import read_image
from torchvision.io import ImageReadMode
from torchvision.transforms import v2
import torchvision.transforms as transforms
import torch
import numpy
from PIL import Image
import cv2

class ImageItem():
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

    def __init__(self, image, right_image, disparity_image, classification_image):
        self._image: str = image
        self._image_r: str = right_image
        self._dipsarity:  str = disparity_image
        self._classification:  str = classification_image

        self.size: int = 512


    def classification(self) -> torch.Tensor:
        gt = read_image(self._classification, mode=ImageReadMode.UNCHANGED)

        _gt = torch.zeros(size=(1, gt.shape[1], gt.shape[2]), dtype=torch.bool)
        _gt[0, :,:] = torch.tensor(0, dtype=torch.bool)

        for i, val in self.movable_labels.items():
            Y, X = numpy.where(gt[0]==torch.tensor(i, dtype=torch.int8))
            
            if val == 1: # movable
                _gt[0, Y, X] = torch.tensor(1, dtype=torch.bool)
        
        

        return v2.Resize(size=self.size, interpolation=v2.InterpolationMode.NEAREST)(_gt)

    def classification_image(self) -> torch.Tensor:
        _ti = self.classification()
        ti = numpy.zeros(shape=(_ti.shape[1], _ti.shape[2], 3) , dtype=numpy.uint8)
        Y, X = numpy.where(_ti[0]==0) # unmovable
        ti[Y, X, :] = (40, 209, 31)
        Y, X = numpy.where(_ti[0]==1) # movable
        ti[Y, X, :] = (194, 17, 73)
        target_image = Image.fromarray(ti, 'RGB')
        transform = transforms.Compose([ 
            transforms.PILToTensor() 
        ]) 
        img_tensor = transform(target_image)
        return img_tensor

    def left(self) -> torch.Tensor:
        return v2.Resize(size=self.size)(read_image(self._image, mode=ImageReadMode.RGB))
    
    def right(self) -> torch.Tensor:
        return v2.Resize(size=self.size)(read_image(self._image, mode=ImageReadMode.RGB))

    def disparity(self) -> torch.Tensor:
        disp = cv2.imread(self._dipsarity, cv2.IMREAD_UNCHANGED)

        image = numpy.zeros(shape=(disp.shape[0], disp.shape[1], 3) , dtype=numpy.uint8)

        ma = disp.max()
        step = (ma + 1) / 255

        for i in range(0, 255):
            s = step * i
            e = step * (i+1)
            X, Y = numpy.where(numpy.logical_and(disp>=s, disp<=e)) # unmovable
            image[X, Y, :] = (i, i, i)

        transform = transforms.Compose([ 
            transforms.PILToTensor() 
        ]) 
        img_tensor = transform(Image.fromarray(image, 'RGB')) 
        return v2.Resize(size=self.size, interpolation=v2.InterpolationMode.NEAREST)(img_tensor)


class CityScapes():

    def __init__(self, dataset_dir):
        self.dataset_dir = dataset_dir

        left_image_path = os.path.join(self.dataset_dir, "leftImg8bit_trainvaltest", "leftImg8bit", "train")
        right_image_path = os.path.join(self.dataset_dir, "rightImg8bit_trainvaltest", "rightImg8bit", "train")
        disp_truth_path = os.path.join(self.dataset_dir, "disparity_trainvaltest", "disparity", "train")
        segm_truth_path = os.path.join(self.dataset_dir, "gtFine_trainvaltest", "gtFine", "train")

        self.images: List[ImageItem] = []

        # loop images
        for city in os.listdir(left_image_path):
            city_dir = os.path.join(left_image_path, city)
            if os.path.isdir(city_dir):
                # print(city)
                for l_image in os.listdir(city_dir):
                    # print(l_image)
                    if l_image.endswith(".png"):
                        # Right image path
                        l_i_p = os.path.join(city_dir, l_image)
                        if not os.path.isfile(l_i_p):
                            raise Exception(f"Left image path {l_i_p} not found")

                        # Left image path
                        r_i_p = os.path.join(right_image_path, city, l_image.replace("leftImg8bit", "rightImg8bit"))
                        if not os.path.isfile(r_i_p):
                            raise Exception(f"Right image path {r_i_p} not found")

                        # Disparity image path
                        d_i_p = os.path.join(disp_truth_path, city, l_image.replace("leftImg8bit", "disparity"))
                        if not os.path.isfile(d_i_p):
                            raise Exception(f"Disparity image path {d_i_p} not found")

                        # Classification image path
                        c_i_p = os.path.join(segm_truth_path, city, l_image.replace("leftImg8bit", "gtFine_labelIds"))
                        if not os.path.isfile(c_i_p):
                            raise Exception(f"Classification image path {c_i_p} not found")

                        image = ImageItem(
                            l_i_p,
                            r_i_p,
                            d_i_p,
                            c_i_p
                            )
                        
                        self.images.append(image)

    def __len__(self):
        return len(self.images)

