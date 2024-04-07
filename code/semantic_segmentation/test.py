import torchvision
import numpy
import torch
import argparse
import segmentation_utils
import cv2
from PIL import Image
from dataset import CityScapesDataset
import time
import tqdm
from torchvision.transforms.functional import to_pil_image

DEVICE = "cuda"

model = torchvision.models.segmentation.fcn_resnet50(weights=None, num_classes=2)
model.load_state_dict(torch.load("model_best.pth"), strict=False)
model.eval()
model.to(DEVICE)


dataset = CityScapesDataset()
loader = torch.utils.data.DataLoader(
    torch.utils.data.Subset(dataset, [1, ]), batch_size=50, shuffle=True, num_workers=20)


for image, target in tqdm.tqdm(loader):

    import sys
    numpy.set_printoptions(threshold=sys.maxsize)

    image, target = image.to(DEVICE), target.to(DEVICE)

    to_pil_image(image[0]).show()

    _ti = target[0].to("cpu").numpy()
    ti = numpy.zeros(shape=(_ti.shape[1], _ti.shape[2], 3) , dtype=numpy.uint8)
    Y, X = numpy.where(_ti[0]==1) # unmovable
    ti[Y, X, :] = (40, 209, 31)
    Y, X = numpy.where(_ti[1]==1) # movable
    ti[Y, X, :] = (194, 17, 73)

    target_image = Image.fromarray(ti, 'RGB')
    target_image.show()



    test = image.to("cuda")
    y_pred = model(test)["out"]
    _ti = y_pred[0].detach().to("cpu").numpy()
    ti = numpy.zeros(shape=(_ti.shape[1], _ti.shape[2], 3) , dtype=numpy.uint8)
    Y, X = numpy.where(_ti[0]>0.9) # unmovable
    # print(_ti[0])
    ti[Y, X, :] = (255, 255, 255)
    Y, X = numpy.where(_ti[1]>0.9) # movable
    ti[Y, X, :] = (255, 0, 0)
    target_image = Image.fromarray(ti, 'RGB')
    target_image.show()
