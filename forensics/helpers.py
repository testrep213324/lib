import re

import cv2


def image_resize(image, width=None, height=None, inter=cv2.INTER_AREA):
    dim = None
    (h, w) = image.shape[:2]

    # if both the width and height are None, then return the
    # original image
    if width is None and height is None:
        return image

    # check to see if the width is None
    if width is None:
        # calculate the ratio of the height and construct the
        # dimensions
        r = height / float(h)
        dim = (int(w * r), height)

    # otherwise, the height is None
    else:
        # calculate the ratio of the width and construct the
        # dimensions
        r = width / float(w)
        dim = (width, int(h * r))

    # resize the image
    resized = cv2.resize(image, dim, interpolation=inter)

    # return the resized image
    return resized


class TextCleaner:
    re_cln_str = re.compile('[^0-9a-zA-Zа-яА-ЯёЁ ,.]+')
    re_cln_space= re.compile(' +')

    def clean_string(self, text, replacement):
        return self.re_cln_str.sub(replacement, text)

    def clean_space(self, text):
        return self.re_cln_space.sub("", text).strip()
