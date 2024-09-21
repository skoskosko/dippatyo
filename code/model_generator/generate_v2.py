






from dataset import CityScapes
from typing import TYPE_CHECKING, List
from matplotlib import pyplot as plt
from torchvision.utils import make_grid
import torchvision.transforms.functional as F
import torchvision.transforms as transforms
from scipy.ndimage import label
import os

from PIL import Image
import torch
from estimator import HorisontalEstimator, VerticalEstimator, EstimatorBase, FocalEstimator


import numpy

# Due to my laziness in setting up a vscode environment
if TYPE_CHECKING:
    from .dataset import CityScapes, ImageItem

def group_coordinates(grid) -> List[numpy.ndarray]:

    groups, nlabels = label(grid)
    response = []
    for i in range(1, nlabels+1):
        _group = numpy.zeros(grid.shape)
        Y, X = numpy.where(groups==i)
        for _i in range(len(Y)):
            _group[Y[_i], X[_i]] = 1
        response.append(_group)

    return response

def show(imgs):
    if not isinstance(imgs, list):
        imgs = [imgs]
    fix, axs = plt.subplots(ncols=len(imgs), squeeze=False)
    for i, img in enumerate(imgs):
        img = img.detach()
        img = F.to_pil_image(img)
        axs[0, i].imshow(numpy.asarray(img))
        axs[0, i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])

def get_points_around(x, y, radius, x_l, y_l):
    points = []
    
    # Loop through all possible points within the bounding square of the radius
    for dx in range(-radius, radius + 1):
        for dy in range(-radius, radius + 1):
            # Calculate the new point
            new_x = int(x + dx)
            new_y = int(y + dy)
            
            # Check if the new point is within the circle (using the radius constraint)
            if (dx**2 + dy**2) <= radius**2:
                # Check if the new point is within the limits
                if 0 <= new_x <= x_l and 0 <= new_y <= y_l:
                    points.append((new_x, new_y))
                    
    return points

def get_line_points(x1, y1, x2, y2, radius=1, x_l = 1024, y_l = 512):
    points_x = [x1]
    points_y = [y1]

    x_m = x2-x1
    y_m = y2-y1

    x = x1
    y = y1
    steps = abs(x_m)+abs(y_m)
    x_s = x_m / steps
    y_s = y_m / steps
    for _i in range(steps):

        x += x_s
        y += y_s

        points_x.append(int(x))
        points_y.append(int(y))
            
    points_x.append(x2)
    points_y.append(y2)

    thick_points_x = []
    thick_points_y = []

    for x, y in zip(points_x, points_y):
        for dx in range(-radius, radius+1):
            for dy in range(-radius, radius+1):
                # Only keep the points within the radius from the center (x, y)
                if dx**2 + dy**2 <= radius**2:
                    new_x = x + dx
                    new_y = y + dy
                    # Ensure the points are within the bounds (optional)
                    if 0 <= new_x < x_l and 0 <= new_y < y_l:
                        thick_points_x.append(new_x)
                        thick_points_y.append(new_y)

    return numpy.array(thick_points_x), numpy.array(thick_points_y)

    # return numpy.array(points_x), numpy.array(points_y)

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
for index, image in enumerate(dataset.images):

    print(f"{image.city} : {image.name}")
    print(f"{index+1}/{len(dataset.images)}")


    def check_estimate(type, city, name):
        path = "/home/esko/Documents/Dippatyo/output"
        return os.path.exists(os.path.join(path, type,city, name))
    
    if check_estimate("focal", image.city, image.name): continue


    classification: torch.Tensor = image.classification_image()
    disparity: torch.Tensor = image.disparity()

    c: torch.Tensor = image.classification()
    depth_reference: numpy.ndarray = torch.Tensor.numpy(image.disparity())
    groups = group_coordinates(torch.Tensor.numpy(c[0, :, :]))

    # Estimate new disparity
    
    focal_estimate = FocalEstimator().estimate(groups, image.disparity())

    def save_estimate(type, city, name, image):
        path = "/home/esko/Documents/Dippatyo/output"
        if not os.path.exists(os.path.join(path, type)):
            os.makedirs(os.path.join(path, type))
        if not os.path.exists(os.path.join(path, type, city)):
            os.makedirs(os.path.join(path, type, city))
        transform = transforms.ToPILImage()
        img = transform(image)
        img.save(os.path.join(path, type, city, name))

    save_estimate("focal", image.city, image.name, focal_estimate)

    def show_image():
        x = int(depth_reference.shape[1]/2.5)
        y = int(depth_reference.shape[2]/2)


        depth_reference[:, x-5:x+5, y-5:y+5] = numpy.array([255, 255, 0])[:, None, None]

        radius = 2
        color = numpy.array([255, 0, 0])[:, None]  # Shape (3, 1)

        points = 5
        for i in range(points):
            x2 = int((depth_reference.shape[2]-1) * (i / (points-1)))
            y2 = int((depth_reference.shape[1]-1) * (i / (points-1)))

            # depth_reference[:, 0:10, x2-5:x2+5] = numpy.array([255, 0, 255])[:, None, None]
            # depth_reference[:, 502:512, x2-5:x2+5] = numpy.array([100, 100, 0])[:, None, None]
            # depth_reference[:, y2-5:y2+5, 0:10] = numpy.array([150, 150, 0])[:, None, None]
            # depth_reference[:, y2-5:y2+5, 1014:1024] = numpy.array([150, 0, 150])[:, None, None]

            line_points = get_line_points(y, x, x2, 0, radius)
            color_repeated = numpy.repeat(color, line_points[0].shape[0], axis=1)
            depth_reference[:, line_points[1],line_points[0]] = color_repeated

            line_points = get_line_points(y, x, x2, 511, radius)
            color_repeated = numpy.repeat(color, line_points[0].shape[0], axis=1)
            depth_reference[:, line_points[1],line_points[0]] = color_repeated

            line_points = get_line_points(y, x, 0, y2, radius)
            color_repeated = numpy.repeat(color, line_points[0].shape[0], axis=1)
            depth_reference[:, line_points[1],line_points[0]] = color_repeated

            line_points = get_line_points(y, x, 1023, y2, radius)
            color_repeated = numpy.repeat(color, line_points[0].shape[0], axis=1)
            depth_reference[:, line_points[1],line_points[0]] = color_repeated

        d: numpy.ndarray = torch.Tensor.numpy(disparity)
        for group in groups:        
            color = list(numpy.random.choice(range(256), size=3))
            Y, X = numpy.where(group==1)
            for i in range(len(Y)):
                d[:, Y[i], X[i]] = color

        grid = make_grid([torch.from_numpy(d), torch.from_numpy(depth_reference), focal_estimate])
        show(grid)
        plt.show()

    # show_image()