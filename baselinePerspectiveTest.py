import cv2
import numpy as np


test_image = cv2.imread("proportion/top-test-3.jpg")

img_hsv = cv2.cvtColor(test_image, cv2.COLOR_BGR2HLS)

img_gray = cv2.cvtColor(test_image, cv2.COLOR_BGR2GRAY)

h = cv2.split(img_hsv)[0]

lower_bound = np.array([260, 0.47, 0])
upper_bound = np.array([310, 1, 0.75])

mask_red_1 = cv2.inRange(h, 0, 4)
mask_red_2 = cv2.inRange(h, 170, 181)

red_mask = cv2.bitwise_or(mask_red_1, mask_red_2)

mask = cv2.inRange(img_hsv, lower_bound, upper_bound)

filtered = cv2.bitwise_and(test_image, test_image, mask=red_mask)

binarized = cv2.threshold(cv2.cvtColor(filtered, cv2.COLOR_BGR2GRAY),0,255,cv2.THRESH_OTSU)[1]

strelErode = cv2.getStructuringElement(cv2.MORPH_RECT, (23,23))
strelDilate = cv2.getStructuringElement(cv2.MORPH_RECT, (23,23))

eroded = cv2.erode(binarized,strelErode)
dilated = cv2.dilate(eroded,strelErode)

(numLabels, labels, boxes, centroids) = cv2.connectedComponentsWithStats(dilated)


for i in range(1, len(boxes)):
    x = boxes[i][0]
    y = boxes[i][1]
    w = boxes[i][2]
    h = boxes[i][3]
    test_image = cv2.rectangle(test_image, (x, y), (x + w, y + h), (0, 255, 0), 2)


camera_sensor_pixels_dims = (8032, 5983) #48MP
camera_sensor_size_dims = (9.8, 7.3) #mm
real_pixel_dims = (0.00122, 0.00122) #mm
camera_focal_length = 6.9 #mm 6.9mm(actual) 24mm equivalent
camera_appeture = 1.8

print(test_image.shape)

print(f"pixel height: {boxes[1][3]} px")
print(f"pixel width: {boxes[1][2]} px \n")

sensor_pixels_height = (boxes[1][3] * camera_sensor_pixels_dims[0]) / test_image.shape[0]
sensor_pixels_width = (boxes[1][2] * camera_sensor_pixels_dims[1]) / test_image.shape[1]

print(f"sensor_pixels_height: {sensor_pixels_height} pixels")
print(f"sensor_pixels_width: {sensor_pixels_width} pixels \n")

sensor_image_height = real_pixel_dims[0]*sensor_pixels_height
sensor_image_width = real_pixel_dims[1]*sensor_pixels_width

print(f"sensor image height: {sensor_image_height} mm")
print(f"sensor image width: {sensor_image_width} mm \n")

inverse_focal_length = 1 / camera_focal_length
print(f"inverse_focal_length: {inverse_focal_length}")

inverse_image_height = 1 / sensor_image_height
print(f"inverse_image_height: {inverse_image_height} \n")


object_height = 1 / abs(1/(camera_focal_length) - 1/(sensor_image_height))
object_width = 1 / abs(1/(camera_focal_length) - 1/(sensor_image_width))

print(f"object height: {object_height} mm \n")
print(f"object width: {object_width} mm \n")

Ca = 35.808 * pow(boxes[1][3], -0.997)

h_camera = 0.72 #cm

h_object = h_camera - Ca

print(f"Ca: {Ca} cm")
print(f"Altura objeto: {h_object} cm")

print(f"Volume do objeto: {h_object * (object_height*0.1) * (object_width*0.1)} cm3")

cv2.imshow("test", test_image)
#cv2.imshow("masked", binarized)
#cv2.imshow("final", dilated)
#cv2.imshow("masking", cv2.bitwise_and(test_image,test_image, mask=dilated))
cv2.waitKey(0)
cv2.destroyAllWindows()