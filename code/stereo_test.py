import cv2
from matplotlib import pyplot as plt
import os
os.environ["QT_QPA_PLATFORM"] = "wayland"


leftImageDir = "/home/eskotakku/Documents/Dippatyo/dataset/leftImg8bit_trainvaltest/leftImg8bit"
leftImagePath = os.path.join(leftImageDir, "train", "aachen", "aachen_000000_000019_leftImg8bit.png")

rightImageDir = "/home/eskotakku/Documents/Dippatyo/dataset/rightImg8bit_trainvaltest/rightImg8bit"
rightImagePath = os.path.join(rightImageDir, "train", "aachen", "aachen_000000_000019_rightImg8bit.png")


imgL = cv2.imread(leftImagePath, cv2.IMREAD_GRAYSCALE)
imgR = cv2.imread(rightImagePath, cv2.IMREAD_GRAYSCALE) # cv2.IMREAD_COLOR
# stereo = cv2.StereoBM.create(numDisparities=128, blockSize=5)
stereo = cv2.StereoSGBM.create(numDisparities=128, blockSize=10, mode=cv2.StereoSGBM_MODE_HH)
# stereo = cv2.StereoSGBM.create(numDisparities=128, blockSize=20)


disparity = stereo.compute(imgL,imgR)


# plt.imshow(imgL,'gray')
# plt.show()
f, axarr = plt.subplots(1,2,  figsize=(30, 10))
axarr[0].imshow(imgL,'gray')
axarr[1].imshow(disparity,'gray')

# plt.imshow(disparity,'gray')
plt.show()