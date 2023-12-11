import cv2
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
#from mpl_toolkits.mplot3d import Axes3D

BLOCK_SIZE = 15
SEARCH_BLOCK_WINDOW = 60
SEARCH_BLOCK_SIZE = 56

FOCAL_LENGTH = 0.0069 #m
T = 0.04 #m

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

    return np.sum(abs(pixel_vals_1 - pixel_vals_2))

def sum_of_sqr_diff(pixel_vals_1, pixel_vals_2):
    """
    Args:
        pixel_vals_1 (numpy.ndarray): pixel block from left image
        pixel_vals_2 (numpy.ndarray): pixel block from right image

    Returns:
        float: Sum of square difference between individual pixels
    """
    if pixel_vals_1.shape != pixel_vals_2.shape:
        return -1

    return np.sum(pow(abs(pixel_vals_1 - pixel_vals_2),2))

def match_blocks(y, left_block, right_blocks_array):
    first = True
    min_sad = None
    min_block = None
    min_x = 0
    x = 0
    for right_block in right_blocks_array:
        sad = sum_of_abs_diff(left_block, right_block)
        if first:
            min_sad = sad
            min_block = right_block
            min_x = x
            first = False
        else:
            if sad < min_sad:
                min_sad = sad
                min_block = right_block
                min_x = x

        x += 1

    #print("correspondent x: ", min_x)
    return min_block

def calculate_disparity(img_left, img_right):
    cpy_left_image = img_left.copy()
    cpy_right_image = img_right.copy()

    x_max = min(img_left.shape[1], img_right.shape[1])
    y_max = min(img_left.shape[0], img_right.shape[0])

    left_blocks = []
    right_blocks = []

    disparity_map = np.zeros((y_max, x_max))

    for y in range(0, y_max, BLOCK_SIZE):
        sub_left_blocks = []
        sub_right_blocks = []
        for x in range(0, x_max, BLOCK_SIZE):
            left_block = (x, y, x + BLOCK_SIZE, y + BLOCK_SIZE)
            right_block = (x, y, x + BLOCK_SIZE, y + BLOCK_SIZE)
            if left_block[2] <= x_max and left_block[3] <= y_max:
                cpy_left_image = cv2.rectangle(cpy_left_image, (x, y), (x + BLOCK_SIZE, y + BLOCK_SIZE), (0, 255, 0), 1)
                sub_left_blocks.append(left_block)

            if right_block[2] <= x_max and right_block[3] <= y_max:
                cpy_right_image = cv2.rectangle(cpy_right_image, (x, y), (x + BLOCK_SIZE, y + BLOCK_SIZE), (0, 255, 0),
                                                1)
                sub_right_blocks.append(right_block)

        left_blocks.append(sub_left_blocks)
        right_blocks.append(sub_right_blocks)

    cv2.imshow("left", cpy_left_image)
    cv2.imshow("right", cpy_right_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    #cv2.imwrite("disparity/block-left-" + str(BLOCK_SIZE) + "_" + str(SEARCH_BLOCK_WINDOW) + ".jpg", cpy_left_image)
    #cv2.imwrite("disparity/block-right-" + str(BLOCK_SIZE) + "_" + str(SEARCH_BLOCK_WINDOW) + ".jpg", cpy_right_image)

    for i in tqdm(range(len(left_blocks))):
        j = 0
        for left_block in left_blocks[i]:
            right_image_valid_blocks = []
            for x in range(max(0, j - SEARCH_BLOCK_WINDOW), len(right_blocks[i])):
                if right_blocks[i][x][0] < left_block[2]:
                    right_valid_block = img_right[right_blocks[i][x][0]:right_blocks[i][x][2],
                                                  right_blocks[i][x][1]:right_blocks[i][x][3]]
                    right_image_valid_blocks.append(right_valid_block)

            # print(len(right_valid_blocks))
            left_image_block = img_left[left_block[0]: left_block[2], left_block[1]: left_block[3]]

            right_block = match_blocks(i, left_image_block, right_image_valid_blocks)
            # print(right_block.shape)
            disparity_map[left_block[0]: left_block[2], left_block[1]: left_block[3]] = abs(
                left_image_block - right_block)

            j += 1

        # i += 1

    return disparity_map

left_image = cv2.imread("disparity/test-left.jpg", cv2.IMREAD_GRAYSCALE)
right_image = cv2.imread("disparity/test-right.jpg", cv2.IMREAD_GRAYSCALE)

disparity_map = calculate_disparity(left_image, right_image)

#cv2.imwrite("disparity/disparity-map-" + str(BLOCK_SIZE) + "_" + str(SEARCH_BLOCK_WINDOW) + ".jpg", disparity_map)

disp_map = cv2.imread("disparity/disparity-map-" + str(BLOCK_SIZE) + "_" + str(SEARCH_BLOCK_WINDOW) + ".jpg", cv2.IMREAD_GRAYSCALE)

heatmap = cv2.applyColorMap(disp_map, cv2.COLORMAP_HSV)

heat_h, heat_s, heat_v = cv2.split(heatmap)

heat_mask_1 = cv2.inRange(heat_h, 0, 50)
heat_mask_2 = cv2.inRange(heat_h, 130, 200)

heat_mask = cv2.bitwise_or(heat_mask_1, heat_mask_2)

heat_mask = cv2.bitwise_not(heat_mask)

filtered = cv2.bitwise_and(heatmap, heatmap, mask=heat_mask)

binarized = cv2.threshold(heat_h,0,360,cv2.THRESH_OTSU)[1]

strel = cv2.getStructuringElement(cv2.MORPH_RECT, (15,15))
strelErode = cv2.getStructuringElement(cv2.MORPH_RECT, (17,17))

binarized = cv2.dilate(binarized,strel)
binarized = cv2.erode(binarized,strelErode)


heat_binarized = cv2.bitwise_and(heatmap, heatmap, mask=binarized)
heat_h = cv2.bitwise_and(heat_h, heat_h, mask=binarized)

disp_map = cv2.bitwise_and(disp_map, disp_map, mask=binarized)
#disparity_map = cv2.bitwise_and(disparity_map, disparity_map, mask=binarized)

points_cloud = []

for y in range(len(disparity_map)):
    for x in range(len(disparity_map[y])):
        z = ((FOCAL_LENGTH * T) / max(disparity_map[y][x],1))
        z = 1/(z*1000)
        if z > 0:
            point = (x, y, z)
            points_cloud.append(point)

x, y, z = zip(*points_cloud)


fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

#point cloud
sc = ax.scatter(x, y, z, c=z, cmap='viridis', s=10)

#triangulation
#ax.plot_trisurf(x, y, z, cmap='viridis', edgecolor='none');

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

ax.view_init(elev=47, azim=-50, roll=0)

plt.show()

'''
cv2.imshow('image', disp_map)
cv2.imshow('heatmap', heatmap)
cv2.imshow('filtered', filtered)
cv2.imshow('binarized', binarized)
cv2.imshow('heat_binarized', heat_binarized)
cv2.imshow('hue', heat_h)
cv2.waitKey(0)
cv2.destroyAllWindows()
'''

#cv2.imwrite("heatmaps/filtered-" + str(BLOCK_SIZE) + "_" + str(SEARCH_BLOCK_WINDOW) + ".jpg", filtered)
#cv2.imwrite("heatmaps/binarized-" + str(BLOCK_SIZE) + "_" + str(SEARCH_BLOCK_WINDOW) + ".jpg", binarized)
#cv2.imwrite("heatmaps/heat_binarized-" + str(BLOCK_SIZE) + "_" + str(SEARCH_BLOCK_WINDOW) + ".jpg", heat_binarized)
#cv2.imwrite("heatmaps/heat_hue-" + str(BLOCK_SIZE) + "_" + str(SEARCH_BLOCK_WINDOW) + ".jpg", heat_h)
#cv2.imwrite("heatmaps/heatmap-" + str(BLOCK_SIZE) + "_" + str(SEARCH_BLOCK_WINDOW) + ".jpg", heatmap)

#cv2.imshow("disparity", disparity_map)
#cv2.waitKey(0)
#cv2.destroyAllWindows()


#h, w, c = left_image.shape
#disparity_map = np.zeros((h, w))
#min_index = compare_blocks(y, x, block_left, right_array, block_size=BLOCK_SIZE)
#disparity_map[y, x] = abs(min_index[1] - x)


'''
for y in tqdm(range(BLOCK_SIZE, h-BLOCK_SIZE)):
        for x in range(BLOCK_SIZE, w-BLOCK_SIZE):
            block_left = input_image[y:y + BLOCK_SIZE,
                                    x:x + BLOCK_SIZE]
            min_index = compare_blocks(y, x, block_left,
                                       right_image,
                                       block_size=BLOCK_SIZE)
            disparity_map[y, x] = abs(min_index[1] - x)
'''

#print(disparity_map)

