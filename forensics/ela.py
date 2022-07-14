import io

import cv2
import numpy as np


def ela(rgb_bitmap, quality=100):
    _rgb_bitmap = rgb_bitmap.copy()
    _, buffer = cv2.imencode('buffer.jpg', _rgb_bitmap, [cv2.IMWRITE_JPEG_QUALITY, quality])
    io_buffer = io.BytesIO(buffer)
    recompressed_img = cv2.imdecode(np.frombuffer(io_buffer.getbuffer(), np.uint8), -1)
    ela_img = recompressed_img.astype(np.float64) - _rgb_bitmap.astype(np.float64)
    return (ela_img ** 2).astype(np.uint8)


if __name__ == "__main__":
    img = cv2.imread('/home/john/Desktop/ufo.cache-1.jpg')
    cv2.imshow("Result", ela(img))
    cv2.waitKey(0)
    cv2.destroyAllWindows()
