
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

labels = [
    [  0,  0,  0 ], # unlabeled  
    [111, 74,  0 ], # dynamic  
    [ 81,  0, 81 ], # ground
    [128, 64,128 ], # road
    [244, 35,232 ], # sidewalk
    [250,170,160 ], # parking
    [230,150,140 ], # rail track
    [ 70, 70, 70 ], # building
    [102,102,156 ], # wall
    [190,153,153 ], # fence
    [180,165,180 ], # guard rail
    [150,100,100 ], # bridge
    [150,120, 90 ], # tunnel
    [153,153,153 ], # pole
    [153,153,153 ], # polegroup
    [250,170, 30 ], # traffic light 
    [220,220,  0 ], # traffic sign  
    [107,142, 35 ], # vegetation    
    [152,251,152 ], # terrain
    [ 70,130,180 ], # sky
    [220, 20, 60 ], # person
    [255,  0,  0 ], # rider
    [  0,  0,142 ], # car
    [  0,  0, 70 ], # truck
    [  0, 60,100 ], # bus
    [  0,  0, 90 ], # caravan
    [  0,  0,110 ], # trailer
    [  0, 80,100 ], # train
    [  0,  0,230 ], # motorcycle
    [119, 11, 32 ], # bicycle
    [  0,  0,142 ]  # license plate 
]

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
                        color = os.path.join(self.truth_root, city, file.replace("_leftImg8bit.png", "_gtFine_color.png"))
                        # color = os.path.join(self.truth_root, city, file.replace("_leftImg8bit.png", "_gtFine_instanceIds.png"))
                        if not os.path.isfile(color):
                            raise Exception(color)
                        self.items.append({"image": f, "gt": color})

    def __len__(self):
        return len(self.items)


    def rgb_to_layer(self, rgb, color, label):
        arr = numpy.zeros(rgb.shape[:2]) ## rgb shape: (h,w,3); arr shape: (h,w)
        # print(rgb)
        # print(color)
        # colors = torch.unique(rgb.view(-1, rgb.size(2)), dim=0).numpy()
        # print(colors)
        numpy.set_printoptions(threshold=sys.maxsize)
        for _x, x in enumerate(rgb):
            for _y, y in enumerate(x):
                pixel = numpy.array(rgb[_x][_y])
                print(rgb[_x][_y].shape)
        raise Exception("STOP")
        arr[numpy.all(rgb == color, axis=-1)] = label
        return arr

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        

        image = read_image(self.items[idx]["image"], mode=ImageReadMode.RGB)
        # gt = read_image(self.items[idx]["gt"], mode=ImageReadMode.RGB)
        gt = cv2.imread(self.items[idx]["gt"])
        # labels_array = numpy.array(labels)

        _gt = torch.zeros(size=(len(labels), image.shape[1], image.shape[2]) )

        for i, val in enumerate(labels):
            # print(val)
            # print(gt.shape)
        
            Y, X = numpy.where(numpy.all(gt==val,axis=2))

            if i == 22:
                print(len(Y))
            _gt[i, Y, X] = 1
            # print(i)
            # print(numpy.sum(_gt[i, Y, X].numpy()))

            # print(Y)
            # print(X)
        #     numpy.where()

        # gt = numpy.all(gt == labels_array[:, None, None], axis=3).astype(int)

        # _gt = torch.zeros(size=(len(labels), image.shape[1], image.shape[2]) )
        # for i, val in enumerate(labels):
        #     for y in range(_gt.shape[1]):
        #         for x in range(_gt.shape[2]):
        #             if val[0] == int(gt[0, y, x]) and val[1] == int(gt[1, y, x]) and val[2] == int(gt[2, y, x]):
        #                 _gt[i, y, x] = 1

        # raise Exception("STOP")

        # mask_out = torch.empty(image.shape[1], image.shape[2], dtype=torch.long)

        # label = 0
        # for k in labels:
        #     self.rgb_to_layer(gt, k, label)


        # print(gt.shape)
        # _image = Image.open(self.items[idx]["image"])
        # _gt = Image.open(self.items[idx]["gt"])


        # transform = transforms.Compose([ 
        #     transforms.PILToTensor() 
        # ]) 
        # image = transform(_image)
        # gt = transform(_gt)


        # image = cv2.imread(self.items[idx]["image"], cv2.IMREAD_UNCHANGED)
        # im_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # image = torch.tensor(im_rgb)
        # 2048x1024
        # print(self.items[idx]["image"])
        image = v2.Resize(size=256)(image)
        gt = v2.Resize(size=256, interpolation=v2.InterpolationMode.NEAREST, antialias=False)(_gt)

        # 1024 x 512
        return image.float(), gt.float()
