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
# model.load_state_dict(torch.load("model_best.pth"), strict=False)

model.eval()

dataset = CityScapesDataset()



loader = torch.utils.data.DataLoader(
    torch.utils.data.Subset(dataset, [1, ]), batch_size=2, shuffle=True, num_workers=0)


# to_pil_image(gt).show()
DEVICE = "cuda"
# model.to(DEVICE)

for image, target in tqdm.tqdm(loader):

    import sys
    numpy.set_printoptions(threshold=sys.maxsize)

    image, target = image.to(DEVICE), target.to(DEVICE)

    to_pil_image(image[0]).show()

    _ti = target[0].to("cpu").numpy()
    ti = numpy.zeros(shape=(_ti.shape[1], _ti.shape[2], 3) , dtype=numpy.uint8)
    for i, color in enumerate(labels):

        
       
        # print(ti[:, :, 0].shape)
        # print(_ti[i].shape)
        if i  == 22:
            print(numpy.sum(_ti[i]))
            print(color)

        Y, X = numpy.where(_ti[i]>0)
        ti[Y, X, :] = color
        # ti[:, :, 1] = _ti[i] * color[1]
        # ti[:, :, 2] = _ti[i] * color[2]
        # if numpy.sum(_ti[i, :, :]) > 0 and i > 2:
        #     print(color)
        #     print(i)
        #     # val = _ti[i, :, :]
        #     print(numpy.sum(_ti[i, :, :]))
            
        #     print(_ti[i, :, :])
            
        #     print(ti)
        #     raise Exception("stop")

    # print(ti)
    # test = torch.from_numpy(ti)
    # print(test.shape)
    # print(image[0])
    # print(ti.shape)
    # print(test)
    # I = cv2.cvtColor(ti, cv2.COLOR_BGR2RGB)
    # target_image = Image.fromarray(ti, 'RGB')
    target_image = Image.fromarray(ti, 'RGB')
    target_image.show()


    # y_pred = model(image)["out"]

    # print(type(y_pred))
    # print(y_pred)
    # to_pil_image(target[0]).show()
    # to_pil_image(y_pred[0]).show()