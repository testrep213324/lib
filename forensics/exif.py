import io
import os
import re

import cv2
import exifread
import numpy as np
from PIL import Image

from forensics.helpers import TextCleaner


class _HeaderParser:
    tag_list = (
        "jfif", "exif", "photoshop", "adobe", "hp", "apple", "canon", "kodak", "finepix", "powershot",
        "leadtechnologies", "nikon", "handmadesoftware", "viewer", "sierra", "photopc", "office",
        "microsoft", "ecosalon", "cyberview", "colorvision", "appl", "motorola", "konica",
        "imagemagick", "photodex", "nokia", "neat", "casio", "olympus", "dumpr", "madewith",
        "sony", "processedby", "photosuite", "roxio", "gimp", "createdwith", "paint", "xatcom",
        "imageoptimizer", "watermark", "hcreator", "acdsystems", "toshiba", "mathematica",
        "wolfram", "xcreator", "intel", "createdby", "imagegear", "accusoft", "icc_profile",
        "nkon", "cano", "compressor", "ticc_profile", "sunplus", "generatedby", "creator",
        "ijg", "samsung", "ajpeg", "http", "encoder", "wiki", "www", "com", "copyright",
        "license", "public", "huawei", "digma", "mediatek", "application", "camera",
        "printim", "fujifilm", "ducky")

    def __init__(self, file_path):
        self.file_path = file_path
        self.string = self.read_string()
        self.stop_tags = self.tag_detector()

    def read_string(self):
        with open(self.file_path, 'rb') as f:
            line = f.read(512)
        data = line.decode('utf-8', 'ignore')
        data = data.split("\n")[0]
        data = re.sub(r'[^A-Za-z0-9!@#$%^&*()_+\n]+', '', data)
        return data.lower()

    def tag_detector(self):
        tags = []
        for tag in self.tag_list:
            if tag in self.string:
                tags.append(tag)
        return tags


class ExifReader:
    def __init__(self, file_path):
        self.file_path = file_path
        self.cln = TextCleaner()

    def check_path(self):
        if not os.path.isfile(self.file_path):
            raise Exception("Warning: image does not exist")

    def exif(self):
        with open(self.file_path, 'rb') as f:
            tags = exifread.process_file(f)
        tags.pop('JPEGThumbnail', None)

        clean_tags = {}
        for key in tags:
            c_key = self.cln.clean_string(key, "")
            c_key = self.cln.clean_space(c_key)

            c_val = self.cln.clean_string(tags[key].printable, "")
            c_val = self.cln.clean_space(c_val)
            clean_tags[c_key] = c_val
        return clean_tags

    def thumbnail(self):
        with open(self.file_path, 'rb') as f:
            tags = exifread.process_file(f)

        if "JPEGThumbnail" in tags:
            buffer = io.BytesIO()
            buffer.write(tags['JPEGThumbnail'])
            buffer.seek(0)
            file_bytes = np.asarray(bytearray(buffer.read()), dtype=np.uint8)
            return cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    def quantization_tables(self):
        fd = Image.open(self.file_path)
        if fd.format not in ("JPG", "JPEG"):
            return
        luminance = fd.quantization.get(0)
        chrominance = fd.quantization.get(1)
        return luminance, chrominance

    def quality(self):
        luminance, chrominance = self.quantization_tables()
        av_tab_y = np.array(luminance).mean()
        av_tab_cb_cr = np.array(chrominance).mean()
        av = (av_tab_y + av_tab_cb_cr + av_tab_cb_cr) / 3
        d = abs(av_tab_y - av_tab_cb_cr) * 0.49 + abs(av_tab_y - av_tab_cb_cr) * 0.49
        return int(100 - av + d)

    def tags(self):
        head = _HeaderParser(self.file_path)
        return head.stop_tags


if __name__ == "__main__":
    pic_path = '/home/john/Desktop/ufo.cache-1.jpg'

    ex = ExifReader(pic_path)
    # ex.check_path()
    #
    # print(ex.exif())
    # # print(ex.thumbnail())
    # print(ex.quantization_tables())
    # print(ex.quality())
    # print(ex.tags())