import cv2
from matplotlib import pyplot as plt
import os
# os.environ["QT_QPA_PLATFORM"] = "wayland"

def highpass(img, sigma):
    return img - cv2.GaussianBlur(img, (0,0), sigma) + 127

leftImageDir = "/home/esko/Documents/Dippatyo/dataset/leftImg8bit_trainvaltest/leftImg8bit"
leftImagePath = os.path.join(leftImageDir, "train", "aachen", "aachen_000000_000019_leftImg8bit.png")

rightImageDir = "/home/esko/Documents/Dippatyo/dataset/rightImg8bit_trainvaltest/rightImg8bit"
rightImagePath = os.path.join(rightImageDir, "train", "aachen", "aachen_000000_000019_rightImg8bit.png")


imgL = cv2.imread(leftImagePath, cv2.IMREAD_GRAYSCALE)
imgR = cv2.imread(rightImagePath, cv2.IMREAD_GRAYSCALE) # cv2.IMREAD_COLOR
truth = cv2.imread("/home/esko/Documents/Dippatyo/dataset/disparity_trainvaltest/disparity/train/aachen/aachen_000000_000019_disparity.png", cv2.IMREAD_GRAYSCALE) # cv2.IMREAD_COLOR

imgL = highpass(imgL, 20)
imgR = highpass(imgR, 20)

# stereo = cv2.StereoBM.create(numDisparities=128, blockSize=5)
# stereo = cv2.StereoSGBM.create(numDisparities=128, blockSize=10, mode=cv2.StereoSGBM_MODE_HH)
stereo = cv2.StereoSGBM.create(numDisparities=128, blockSize=20, mode=cv2.StereoSGBM_MODE_HH)


disparity = stereo.compute(imgL,imgR)


# plt.imshow(imgL,'gray')
# plt.show()
f, axarr = plt.subplots(1,2,  figsize=(30, 10))
axarr[0].imshow(truth,'gray')
axarr[1].imshow(disparity,'gray')

# plt.imshow(disparity,'gray')
plt.show()
