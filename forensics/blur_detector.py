import cv2
import numpy as np


def detect_blur(cv_bitmap_grayscale, window=32):
    local_cv_bitmap_grayscale = cv_bitmap_grayscale.copy()
    img_height, img_width = local_cv_bitmap_grayscale.shape[:2]
    img_block_blurred = np.zeros_like(local_cv_bitmap_grayscale)

    for x in range(0, img_width, window):
        for y in range(0, img_height, window):
            crop_img = local_cv_bitmap_grayscale[y:y + window, x:x + window]
            val = cv2.Laplacian(crop_img, cv2.CV_64F).var()
            img_block_blurred[y:y + window, x:x + window] = val
    return img_block_blurred.astype(np.uint8)


if __name__ == "__main__":
    pic = cv2.imread(r'/home/john/Desktop/ufo.cache-1.jpg', 0)
    blur_bitmap = detect_blur(pic, window=8)
    cv2.imshow("blur_bitmap", blur_bitmap)
    cv2.waitKey(0)