# Extract adaptive histogram for feature description.
# See http://openaccess.thecvf.com/content_cvpr_2018_workshops/papers/w3/Tang_Single-Camera_and_Inter-Camera_CVPR_2018_paper.pdf for detail.

#from skimage.feature import hog
import numpy as np
import math
import cv2
import sys
import time

def bhattacharyya(a, b):
    if not len(a) == len(b):
        raise ValueError("a and b must be of the same size")
    return -math.log(sum((math.sqrt(u * w) for u, w in zip(a, b))))

def adaptive_hist(image):
    """
    image is opencv image format. I.e. image = cv2.imread(path).
    """
    mask = np.zeros(image.shape[:2], np.uint8)
    # spatially weighted by Gaussian distribtuion?
    mask = cv2.ellipse(mask, (image.shape[1] // 2,image.shape[0] // 2),
            (image.shape[1] // 2,image.shape[0] // 2), 0, 0, 360, 255, -1)

    # RGB color histogram
    hist1 = cv2.calcHist([image], [0], mask, [16], [0, 256]).reshape(1, -1)
    hist2 = cv2.calcHist([image], [1], mask, [16], [0, 256]).reshape(1, -1)
    hist3 = cv2.calcHist([image], [2], mask, [16], [0, 256]).reshape(1, -1)
    rgb_hist = np.concatenate((hist1, hist2, hist3), axis=1)
    cv2.normalize(rgb_hist, rgb_hist)

    # HSV color histogram
    img_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hist1 = cv2.calcHist([img_hsv], [0], mask, [16], [0, 256]).reshape(1, -1)
    hist2 = cv2.calcHist([img_hsv], [1], mask, [16], [0, 256]).reshape(1, -1)
    hsv_hist = np.concatenate((hist1, hist2), axis=1)
    cv2.normalize(hsv_hist, hsv_hist)

    # YCrCb color histogram
    img_YCrCb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
    hist1 = cv2.calcHist([img_YCrCb], [1], mask, [16], [0, 256]).reshape(1, -1)
    hist2 = cv2.calcHist([img_YCrCb], [2], mask, [16], [0, 256]).reshape(1, -1)
    YCrCb_hist = np.concatenate((hist1, hist2), axis=1)
    cv2.normalize(YCrCb_hist, YCrCb_hist)

    # Lab color histogram
    img_lab = cv2.cvtColor(image, cv2.COLOR_BGR2Lab)
    hist1 = cv2.calcHist([img_lab], [1], mask, [16], [0, 256]).reshape(1, -1)
    hist2 = cv2.calcHist([img_lab], [2], mask, [16], [0, 256]).reshape(1, -1)
    lab_hist = np.concatenate((hist1, hist2), axis=1)
    cv2.normalize(lab_hist, lab_hist)

    # Hog
    #image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    #image_gray = cv2.resize(image_gray, (200,200))
    #hog_hist = hog(image_gray, orientations=8, block_norm = 'L2-Hys', pixels_per_cell=(50,50), cells_per_block=(1,1), visualize=False).reshape(1, -1)
    #cv2.normalize(hog_hist, hog_hist)

    # type?
    #type_hist = np.zeros(8).reshape(1,8) + 0.5
    #type_hist[0, int(image_path[-5])] = 1
    #cv2.normalize(type_hist, type_hist)

    #thist = np.transpose(np.concatenate((3 * rgb_hist, hsv_hist, YCrCb_hist, lab_hist, hog_hist), axis=1))
    thist = np.transpose(np.concatenate((3 * rgb_hist, hsv_hist, YCrCb_hist, lab_hist), axis=1))
    thist = thist / sum(thist)

    return np.transpose(thist)[0]

if __name__ == '__main__':

    image1 = cv2.imread(sys.argv[1])
    image2 = cv2.imread(sys.argv[2])

    stime = time.time()
    hist2 = adaptive_hist(image2)
    etime = time.time()
    print('Extract historgram costs: %f' % (etime-stime))

    hist1 = adaptive_hist(image1)
    etime2 = time.time()
    print('Extract historgram costs: %f' % (etime2-etime))

    print(bhattacharyya(hist1, hist2))
    ptime = time.time()
    print('Calculate distance costs: %f' % (ptime-etime2))
