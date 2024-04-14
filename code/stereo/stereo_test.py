import cv2
from matplotlib import pyplot as plt
import os
import numpy
# os.environ["QT_QPA_PLATFORM"] = "wayland"

def smooth(img):
    kernel = numpy.ones((5,5),numpy.float32)/25
    dst = cv2.filter2D(img,-1,kernel)
    return dst

leftImageDir = "/home/esko/Documents/Dippatyo/dataset/leftImg8bit_trainvaltest/leftImg8bit"
leftImagePath = os.path.join(leftImageDir, "train", "aachen", "aachen_000000_000019_leftImg8bit.png")

rightImageDir = "/home/esko/Documents/Dippatyo/dataset/rightImg8bit_trainvaltest/rightImg8bit"
rightImagePath = os.path.join(rightImageDir, "train", "aachen", "aachen_000000_000019_rightImg8bit.png")


imgL = cv2.imread(leftImagePath, cv2.IMREAD_GRAYSCALE)
imgR = cv2.imread(rightImagePath, cv2.IMREAD_GRAYSCALE) # cv2.IMREAD_COLOR
truth = cv2.imread("/home/esko/Documents/Dippatyo/dataset/disparity_trainvaltest/disparity/train/aachen/aachen_000000_000019_disparity.png", cv2.IMREAD_GRAYSCALE) # cv2.IMREAD_COLOR

# imgL = highpass(imgL, 5)
# imgR = highpass(imgR, 5)

# stereo = cv2.StereoBM.create(numDisparities=128, blockSize=5)
# stereo = cv2.StereoSGBM.create(numDisparities=128, blockSize=5, mode=cv2.StereoSGBM_MODE_HH)
# stereo = cv2.StereoSGBM.create(numDisparities=128, blockSize=20, mode=cv2.StereoSGBM_MODE_HH)
# stereo = cv2.StereoSGBM.create(numDisparities=128, blockSize=5, mode=cv2.StereoSGBM_MODE_HH, P1=1, P2=200)
stereo = cv2.StereoSGBM.create(numDisparities=128, blockSize=5, mode=cv2.StereoSGBM_MODE_HH, P1=20, P2=200)



disparity = stereo.compute(imgL,imgR)

# disparity = smooth(disparity)
# plt.imshow(imgL,'gray')
# plt.show()
f, axarr = plt.subplots(1,2,  figsize=(30, 10))
axarr[0].imshow(truth,'gray')
axarr[1].imshow(disparity,'gray')

# plt.imshow(disparity,'gray')
plt.show()
