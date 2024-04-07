import cv2
from matplotlib import pyplot as plt
import os
import numpy
import tqdm
os.environ["QT_QPA_PLATFORM"] = "wayland"


def sum_of_abs_diff(pixel_vals_1, pixel_vals_2):
    """
    Args:
        pixel_vals_1 (numpy.ndarray): pixel block from left image
        pixel_vals_2 (numpy.ndarray): pixel block from right image

    Returns:
        float: Sum of absolute difference between individual pixels
    """
    if pixel_vals_1.shape != pixel_vals_2.shape:
        return -1

    return numpy.sum(abs(pixel_vals_1 - pixel_vals_2))


def compare_blocks(y, x, block_left, right_array, block_height, block_width):
    """
    Compare left block of pixels with multiple blocks from the right
    image using SEARCH_BLOCK_SIZE to constrain the search in the right
    image.

    Args:
        y (int): row index of the left block
        x (int): column index of the left block
        block_left (numpy.ndarray): containing pixel values within the 
                    block selected from the left image
        right_array (numpy.ndarray]): containing pixel values for the 
                     entrire right image
        block_height (int): Height of block
        block_width (int): Width of block

    Returns:
        tuple: (y, x) row and column index of the best matching block 
                in the right image
    """
    # Get search range for the right image
    x_min = max(0, x - SEARCH_BLOCK_SIZE)
    x_max = min(right_array.shape[1], x + SEARCH_BLOCK_SIZE)
    #print(f'search bounding box: ({y, x_min}, ({y, x_max}))')
    first = True
    min_sad = None
    min_index = None
    for x in range(x_min, x_max):
        block_right = right_array[y: y+block_height,
                                  x: x+block_width]
        sad = sum_of_abs_diff(block_left, block_right)
        #print(f'sad: {sad}, {y, x}')
        if first:
            min_sad = sad
            min_index = (y, x)
            first = False
        else:
            if sad < min_sad:
                min_sad = sad
                min_index = (y, x)

    return min_index

BLOCK_HEIGHT = 3
BLOCK_WIDTH = 10
SEARCH_BLOCK_SIZE = 20

leftImageDir = "/home/eskotakku/Documents/Dippatyo/dataset/leftImg8bit_trainvaltest/leftImg8bit"
leftImagePath = os.path.join(leftImageDir, "train", "aachen", "aachen_000000_000019_leftImg8bit.png")

rightImageDir = "/home/eskotakku/Documents/Dippatyo/dataset/rightImg8bit_trainvaltest/rightImg8bit"
rightImagePath = os.path.join(rightImageDir, "train", "aachen", "aachen_000000_000019_rightImg8bit.png")



left_array = cv2.imread(leftImagePath, cv2.IMREAD_GRAYSCALE)
# h, w = left_array.shape
# left_array = cv2.resize(left_array, dsize=(int(w/4), int(h/4)), interpolation=cv2.INTER_CUBIC)
right_array = cv2.imread(rightImagePath, cv2.IMREAD_GRAYSCALE)
# right_array = cv2.resize(right_array, dsize=(int(w/4), int(h/4)), interpolation=cv2.INTER_CUBIC)

h, w = left_array.shape
disparity_map = numpy.zeros((h, w))

for y in tqdm.tqdm(range(BLOCK_HEIGHT, h-BLOCK_HEIGHT)):
    for x in range(BLOCK_WIDTH, w-BLOCK_WIDTH):
        # print("row")
        block_left = left_array[y:y + BLOCK_HEIGHT,
                                x:x + BLOCK_WIDTH]
        # min_index = (0,0)
        min_index = compare_blocks(y, x, block_left,
                                    right_array,
                                    block_height=BLOCK_HEIGHT,
                                    block_width=BLOCK_WIDTH)
        disparity_map[y, x] = abs(min_index[1] - x)

plt.imshow(disparity_map,'gray')
plt.show()