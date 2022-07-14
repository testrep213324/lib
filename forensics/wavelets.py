import cv2
import numpy as np
import pywt


def detect_wwt(bgr_bitmap, percent=0.02, level=4):  # 0.02, 0.004
    gray_mean = np.mean(bgr_bitmap, -1)

    cof = pywt.wavedec2(gray_mean, wavelet='db1', level=level)
    cof_arr, cof_slices = pywt.coeffs_to_array(cof)
    cof_sorted = np.sort(np.abs(cof_arr.reshape(-1)))

    thresh = cof_sorted[int(np.floor((1 - percent) * len(cof_sorted)))]
    ind = np.abs(cof_arr) > thresh
    cof_filter = pywt.array_to_coeffs(cof_arr * ind, cof_slices, output_format='wavedec2')
    temp = pywt.waverec2(cof_filter, wavelet='db1').astype(np.uint8).copy()

    h, w = gray_mean.shape[:2]
    temp = temp[0:0 + h, 0:0 + w]
    rez = temp - gray_mean.copy()
    rez = rez - rez.min() + 1
    rez = rez * (255.0 / rez.max())
    return cv2.equalizeHist(rez.astype(np.uint8))


if __name__ == "__main__":
    rgb_bitmap = cv2.imread('/home/john/Desktop/ufo.cache-1.jpg')
    cv2.imshow("detect_wwt", detect_wwt(rgb_bitmap))
    cv2.waitKey(0)
