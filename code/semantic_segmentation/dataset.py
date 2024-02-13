
import torch
import os
from torchvision.io import read_image
from torchvision.io import ImageReadMode
from torchvision.transforms import v2

labels = [
    (  0,  0,  0), # unlabeled  
    (111, 74,  0), # dynamic  
    ( 81,  0, 81), # ground
    (128, 64,128), # road
    (244, 35,232), # sidewalk
    (250,170,160), # parking
    (230,150,140), # rail track
    ( 70, 70, 70), # building
    (102,102,156), # wall
    (190,153,153), # fence
    (180,165,180), # guard rail
    (150,100,100), # bridge
    (150,120, 90), # tunnel
    (153,153,153), # pole
    (153,153,153), # polegroup
    (250,170, 30), # traffic light 
    (220,220,  0), # traffic sign  
    (107,142, 35), # vegetation    
    (152,251,152), # terrain
    ( 70,130,180), # sky
    (220, 20, 60), # person
    (255,  0,  0), # rider
    (  0,  0,142), # car
    (  0,  0, 70), # truck
    (  0, 60,100), # bus
    (  0,  0, 90), # caravan
    (  0,  0,110), # trailer
    (  0, 80,100), # train
    (  0,  0,230), # motorcycle
    (119, 11, 32), # bicycle
    (  0,  0,142)  # license plate 
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
                        if not os.path.isfile(color):
                            raise Exception(color)
                        self.items.append({"image": f, "gt": color})

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        image = read_image(self.items[idx]["image"], mode=ImageReadMode.RGB)
        gt = read_image(self.items[idx]["gt"], mode=ImageReadMode.RGB )
        # 2048x1024
        image = v2.Resize(size=512)(image)
        gt = v2.Resize(size=512)(gt)
        # 1024 x 512
        return image.float(), gt.float()
