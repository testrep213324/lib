import cv2
import numpy as np
from scipy import signal


def detect_median_noise(bgr_bitmap, kernel_size=3, multiplier=10):
    local_cv_bgr_bitmap = bgr_bitmap.copy()
    w, h = local_cv_bgr_bitmap.shape[:2]
    image_med = np.zeros((w, h, 3))
    for channel in [0, 1, 2]:
        median = signal.medfilt2d(local_cv_bgr_bitmap[:, :, channel], [kernel_size, kernel_size])
        image_med[:, :, channel] = median
    return abs(local_cv_bgr_bitmap - image_med) * multiplier


if __name__ == "__main__":
    cv_bgr_bitmap = cv2.imread('/home/john/Desktop/ufo.cache-1.jpg')
    cv2.imshow("orig", detect_median_noise(cv_bgr_bitmap, kernel_size=3, multiplier=10))
    cv2.waitKey(0)