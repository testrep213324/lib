import cv2
import numpy as np


def detect_gradient(gray_bitmap):
    _gray_bitmap = gray_bitmap.copy()
    sobel_x = cv2.Sobel(_gray_bitmap, cv2.CV_32F, 1, 0)
    sobel_y = cv2.Sobel(_gray_bitmap, cv2.CV_32F, 0, 1)

    angle = np.arctan2(sobel_x, sobel_y)
    b = np.sqrt(sobel_x ** 2 + sobel_y ** 2)
    g = -np.sin(angle) / 2.0 + 0.5
    r = -np.cos(angle) / 2.0 + 0.5
    result = cv2.merge((b, g, r))
    return (result * 255).astype(np.uint8)


if __name__ == "__main__":
    img_gray = cv2.imread('/home/john/Desktop/ufo.cache-1.jpg', 0)
    cv2.imshow("orig", detect_gradient(img_gray))
    cv2.waitKey()
