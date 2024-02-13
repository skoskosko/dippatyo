import torchvision
import numpy
import torch
import argparse
import segmentation_utils
import cv2
from PIL import Image
from dataset import CityScapesDataset, labels
import time
import tqdm
from torchvision.transforms.functional import to_pil_image


model = torchvision.models.segmentation.fcn_resnet50(pretrained=False, num_classes=3)
model.load_state_dict(torch.load("model_best.pth"), strict=False)

model.eval()

dataset = CityScapesDataset()



loader = torch.utils.data.DataLoader(
    torch.utils.data.Subset(dataset, [1, ]), batch_size=2, shuffle=True, num_workers=0)


# to_pil_image(gt).show()
DEVICE = "cuda"
model.to(DEVICE)

for image, target in tqdm.tqdm(loader):

    image, target = image.to(DEVICE), target.to(DEVICE)

    to_pil_image(image[0]).show()


    y_pred = model(image)["out"]

    print(type(y_pred))
    print(y_pred)
    to_pil_image(y_pred[0]).show()