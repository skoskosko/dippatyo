






from dataset import CityScapes
from typing import TYPE_CHECKING, List
from matplotlib import pyplot as plt
from torchvision.utils import make_grid
import torchvision.transforms.functional as F
import torchvision.transforms as transforms
import os

from PIL import Image
import torch
from estimator import HorisontalEstimator, VerticalEstimator, EstimatorBase


import numpy

# Due to my laziness in setting up a vscode environment
if TYPE_CHECKING:
    from .dataset import CityScapes, ImageItem

def in_group(coordinate, group) -> bool:
    directions = [(-1, 0), (0, -1),(-1, -1), (-1, 1)]
    for d in directions:
        y = coordinate[0] + d[0]
        x = coordinate[1] + d[1]
        if group[y, x] >= 1:
            return True
    return False

def group_coordinates(grid) -> List[numpy.ndarray]:
    groups = []
    Y, X = numpy.where(grid==1) # unmovable
    for i in range(len(Y)):
        iy = Y[i]
        ix = X[i]
        if grid[iy, ix] == 1:
            in_groups = []
            for i, group in enumerate(groups):
                if in_group([iy, ix], group):
                    in_groups.append(i)

            if len(in_groups):
                in_groups.sort(reverse=True)
                _group = numpy.zeros(grid.shape)
                for i in in_groups:
                    _group = _group + groups[i]
                    groups.pop(i)
                _group[iy, ix] = 1
                groups.append(_group)
            else:
                _group = numpy.zeros(grid.shape)
                _group[iy, ix] = 1
                groups.append(_group)

    return groups

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

    classification: torch.Tensor = image.classification_image()
    disparity: torch.Tensor = image.disparity()

    c: torch.Tensor = image.classification()
    d: numpy.ndarray = torch.Tensor.numpy(disparity)
    groups = group_coordinates(torch.Tensor.numpy(c[0, :, :]))
    
    for group in groups:        
        color = list(numpy.random.choice(range(256), size=3))
        Y, X = numpy.where(group==1)
        for i in range(len(Y)):
            d[:, Y[i], X[i]] = color

    # Estimate new disparity
    
    h_estimate = HorisontalEstimator().estimate(groups, disparity)
    v_estimate = VerticalEstimator().estimate(groups, disparity)
    c_estimate = EstimatorBase().combine_estimators(groups, h_estimate, v_estimate)

    h_50_estimate = HorisontalEstimator().estimate(groups, disparity, b=50)
    v_50_estimate = VerticalEstimator().estimate(groups, disparity, b=50)
    c_50_estimate = EstimatorBase().combine_estimators(groups, h_50_estimate, v_50_estimate)

    

    def save_estimate(type, city, name, image):
        path = "/home/esko/Documents/Dippatyo/output"
        if not os.path.exists(os.path.join(path, type)):
            os.makedirs(os.path.join(path, type))
        if not os.path.exists(os.path.join(path, type, city)):
            os.makedirs(os.path.join(path, type, city))
        transform = transforms.ToPILImage()
        img = transform(image)
        img.save(os.path.join(path, type, city, name))


    save_estimate("horisontal", image.city, image.name, h_estimate)
    save_estimate("vertical", image.city, image.name, v_estimate)
    save_estimate("combined", image.city, image.name, c_estimate)

    save_estimate("horisontal_50", image.city, image.name, h_50_estimate)
    save_estimate("vertical_50", image.city, image.name, v_50_estimate)
    save_estimate("combined_50", image.city, image.name, c_50_estimate)

    print(f"{image.city} : {image.name}")
    # grid = make_grid([image.left(), image.classification_image(), image.disparity(), torch.from_numpy(d), h_estimate, v_estimate, c_estimate])

    # show(grid)

    # plt.show()
