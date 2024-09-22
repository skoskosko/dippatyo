
import torch
import random
import math
import torchvision
import time
import tqdm
import numpy
from torchvision.transforms.functional import to_pil_image

torch.cuda.empty_cache()

EPOCHES = 100
BATCH_SIZE = 30
DEVICE = 'cuda'


model = torchvision.models.segmentation.fcn_resnet50(weights=None, num_classes=128)
model.load_state_dict(torch.load("tmp_best_model.pth"), strict=False)
model.eval()
model.to(DEVICE)


def to_img(image):
    img = image.to("cpu")
    _output = torch.zeros(size=(1, img.shape[1], img.shape[2]), dtype=torch.uint8)
    for i in range(128):
        Y, X = numpy.where(img[i] > 0.5)
        # print(len(Y))
        _output[0,Y,X] = i*2
    return _output


# Read external image
from torchvision.transforms import v2
from torchvision.io import read_image
from torchvision.io import ImageReadMode
l = v2.Resize(size=256, interpolation=v2.InterpolationMode.NEAREST, antialias=False)(read_image("/home/esko/Documents/Dippatyo/masters-thesis/figures/outside_image_l.png", mode=ImageReadMode.GRAY))
r = v2.Resize(size=256, interpolation=v2.InterpolationMode.NEAREST, antialias=False)(read_image("/home/esko/Documents/Dippatyo/masters-thesis/figures/outside_image_r.png", mode=ImageReadMode.GRAY))
image = torch.zeros(size=(3, l.shape[1], l.shape[2]), dtype=torch.uint8)
image[0, :, :] = l
image[1, :, :] = r


to_pil_image(l).show()

_image = torch.zeros(size=(1, 3, image.shape[1], image.shape[2]), dtype=torch.uint8)
_image[0, :, :, : ] = image
_image = _image.float().to(DEVICE)

prediction = model(_image)["out"]
to_pil_image(to_img(prediction[0])).show()