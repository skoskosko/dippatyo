






from dataset import CityScapes
from typing import TYPE_CHECKING
from matplotlib import pyplot as plt
from torchvision.utils import make_grid
import torchvision.transforms.functional as F

import numpy

# Due to my laziness in setting up a vscode environment
if TYPE_CHECKING:
    from .dataset import CityScapes, ImageItem

def show(imgs):
    if not isinstance(imgs, list):
        imgs = [imgs]
    fix, axs = plt.subplots(ncols=len(imgs), squeeze=False)
    for i, img in enumerate(imgs):
        img = img.detach()
        img = F.to_pil_image(img)
        axs[0, i].imshow(numpy.asarray(img))
        axs[0, i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])


# Get images into a dataset
    # image should have original
    # Area of movable objects
    # Disparity map
dataset: "CityScapes" = CityScapes("/home/esko/Documents/Dippatyo/dataset")


# Go trough images
    # Go trough images 
    # Remove areas with movable
    # assume area same as around it
    # Ask image to be approved
    # save if approved
for image in dataset.images:
    # print(image._dipsarity)
    # image.left(), image.classification_image(),
    grid = make_grid([image.disparity()])
    show(grid)

    plt.show()
