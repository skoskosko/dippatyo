import torchvision.transforms as vision_transforms
import cv2
import numpy
import torch
from label_color_map import label_color_map as label_map


transform = vision_transforms.Compose([
    vision_transforms.ToTensor(),
    vision_transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])


def get_segment_labels(image, model, device):
    image = transform(image).to(device)
    image = image.unsqueeze(0) # add a batch dimension
    outputs = model(image)
    return outputs

def draw_segmentation_map(outputs):
    labels = torch.argmax(outputs.squeeze(), dim=0).detach().cpu().numpy()
    red_map = numpy.zeros_like(labels).astype(numpy.uint8)
    green_map = numpy.zeros_like(labels).astype(numpy.uint8)
    blue_map = numpy.zeros_like(labels).astype(numpy.uint8)
    
    for label_num in range(0, len(label_map)):
        index = labels == label_num
        red_map[index] = numpy.array(label_map)[label_num, 0]
        green_map[index] = numpy.array(label_map)[label_num, 1]
        blue_map[index] = numpy.array(label_map)[label_num, 2]
        
    segmented_image = numpy.stack([red_map, green_map, blue_map], axis=2)
    return segmented_image

def image_overlay(image, segmented_image):
    alpha = 0.6 # how much transparency to apply
    beta = 1 - alpha # alpha + beta should equal 1
    gamma = 0 # scalar added to each sum
    image = np.array(image)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    segmented_image = cv2.cvtColor(segmented_image, cv2.COLOR_RGB2BGR)
    cv2.addWeighted(segmented_image, alpha, image, beta, gamma, image)
    return image