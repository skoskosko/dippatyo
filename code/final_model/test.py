

# Hae datasetisä hyväksyttävät totuudet

# Hae datasetistä stereokuvat

# Aseta niille joku filteri smoothing tms

# Tee training ja jotain setit

# Treenaa vähä

import torch
from dataset import CityScapesDataset
import random
import math
import torchvision
import time
import tqdm
import numpy
from torchvision.transforms.functional import to_pil_image

torch.cuda.empty_cache()

dataset = CityScapesDataset()


EPOCHES = 100
BATCH_SIZE = 30
DEVICE = 'cuda'


model = torchvision.models.segmentation.fcn_resnet50(weights=None, num_classes=128)
model.load_state_dict(torch.load("model.pth"), strict=False)
model.eval()
model.to(DEVICE)


# loader = torch.utils.data.DataLoader(
#     torch.utils.data.Subset(dataset, [1, ]), batch_size=1, shuffle=True, num_workers=1)

# print(loader[0])

# for image, target in tqdm.tqdm(loader):
index = random.randint(0, len(dataset))
print(index)
image, target = dataset[index]
to_pil_image(dataset.l_image(index)).show()



def to_img(image):
    img = image.to("cpu")
    _output = torch.zeros(size=(1, img.shape[1], img.shape[2]), dtype=torch.uint8)
    for i in range(128):
        Y, X = numpy.where(img[i] > 0.9)
        print(len(Y))
        _output[0,Y,X] = i*2
    return _output



to_pil_image(to_img(target)).show()

_image = torch.zeros(size=(1, 3, image.shape[1], image.shape[2]), dtype=torch.uint8)

_image[0, :, :, : ] = image
_image = _image.float().to(DEVICE)

prediction = model(_image)["out"]
to_pil_image(to_img(prediction[0])).show()