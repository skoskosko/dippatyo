# import cityscapesscripts

# from PIL import Image

import cv2
import os 
from PIL import Image
import numpy
os.environ["QT_QPA_PLATFORM"] = "xcb" # ignoring wayland!! HOW SAD!!


from cityscapesscripts.helpers.version import version as VERSION

# annotation helpers
# from cityscapesscripts.helpers.annotation import Annotation, CsObjectType
# from cityscapesscripts.helpers.labels import name2label, assureSingleInstanceName
# from cityscapesscripts.helpers.labels_cityPersons import name2labelCp
# from cityscapesscripts.helpers.box3dImageTransform import Box3dImageTransform


# Get image 


gtFinePath = "/home/eskotakku/Documents/Dippatyo/dataset/gtFine_trainvaltest/gtFine"
leftImagePath = "/home/eskotakku/Documents/Dippatyo/dataset/leftImg8bit_trainvaltest/leftImg8bit"

 
# Get images

imagePath = os.path.join(leftImagePath, "train", "aachen", "aachen_000000_000019_leftImg8bit.png")

image = cv2.imread(imagePath, cv2.IMREAD_COLOR)



# Load labels


colorPath = os.path.join(gtFinePath, "train", "aachen", "aachen_000000_000019_gtFine_color.png")
# rects = os.path.join(leftImagePath, "test", "berlin", "berlin_000000_000019_leftImg8bit.png")

colors = cv2.imread(colorPath, cv2.IMREAD_COLOR)


added_image = cv2.addWeighted(image,0.9,colors,0.2,0)

im_pil = Image.fromarray(added_image)
im_pil.show()
