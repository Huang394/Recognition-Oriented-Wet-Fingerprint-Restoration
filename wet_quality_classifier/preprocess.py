import numpy as np
import glob
import os
import cv2
import matplotlib.pyplot as plt
plt.switch_backend('agg')

contour_area_limit = 200

img_path = fr'D:\remi\dataset\all_dataset\noise\85\0007_tina_ft_0007_0_786_0_0.bmp'

def find_contour(img):
    ret, thresh = cv2.threshold(img, 245, 255, cv2.THRESH_BINARY)
    contours, hierarchy = cv2.findContours(
        thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours

def find_large_background_area(img):
    # find large background area contour
    contours = find_contour(img)
    mask_img = np.zeros((img.shape[0], img.shape[1]), np.uint8)
    for c in contours:
        contour_area = cv2.contourArea(c)
        if(contour_area > contour_area_limit):
            cv2.drawContours(mask_img, [c], -1, (255),
                             cv2.FILLED)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    erosion_img = cv2.erode(mask_img, kernel)
    # background_img = cv2.bitwise_and(img, erosion_img)

    # contour location
    y, x = np.where(erosion_img == 255)
    large_background_loc = np.column_stack((y, x))
    
    # contour area
    hist = cv2.calcHist([erosion_img], [0], None, [256], [0, 256])
    hist = hist.reshape((256))
    white_value = int(hist[255])

    return erosion_img, large_background_loc, white_value

if __name__ == "__main__":
    origin_img = cv2.imread(img_path)
    gray_img = cv2.cvtColor(origin_img, cv2.COLOR_BGR2GRAY)
    erosion_img, large_background_loc, white_value = find_large_background_area(gray_img)