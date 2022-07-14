import cv2
import numpy as np
from skimage.draw import disk


class Mask:
    def __init__(self, h, w):
        self.h = h
        self.w = w
        self.center = (self.h // 2, self.w // 2)

    def low_pass_filter(self, outer_radius):
        mask = np.zeros((self.h, self.w), dtype=np.double)
        rr, cc = disk(center=self.center, radius=outer_radius, shape=(self.h, self.w))
        mask[rr, cc] = 1
        return mask

    def high_pass_filter(self, inner_radius):
        mask = np.ones((self.h, self.w), dtype=np.double)
        rr, cc = disk(center=self.center, radius=inner_radius, shape=(self.h, self.w))
        mask[rr, cc] = 0
        return mask

    def band_pass_filter(self, outer_radius, inner_radius):
        low_pass_mask = self.low_pass_filter(outer_radius)
        high_pass_mask = self.high_pass_filter(inner_radius)
        return low_pass_mask * high_pass_mask


class Bandpass:
    def __init__(self, grey_bitmap):
        self.grey_bitmap = grey_bitmap.copy()
        self.h, self.w = self.grey_bitmap.shape[:2]

        # min_demention = min(self.h , self.w)
        # if outer_radius is None:
        #     outer_radius = min_demention // 1.5
        #
        # if inner_radius is None:
        #     inner_radius = min_demention // 5

        self.mask = Mask(self.h, self.w)

    @staticmethod
    def normalize(de_centered_spectrum):
        # normalize
        filtered_img = np.fft.ifft2(de_centered_spectrum)
        logy_filtered = np.log(1 + np.abs(filtered_img.real))
        logy_filtered -= logy_filtered.min()
        bitmap = (logy_filtered * 255.0 / logy_filtered.max()).astype(np.uint8)
        return bitmap

    def band_pass(self, outer_radius, inner_radius):
        bandpass_mask = self.mask.band_pass_filter(outer_radius=outer_radius, inner_radius=inner_radius)
        original_spectrum = np.fft.fft2(self.grey_bitmap)
        centered_spectrum = np.fft.fftshift(original_spectrum)
        mask_applied = bandpass_mask * centered_spectrum
        de_centered_spectrum = np.fft.ifftshift(mask_applied)
        return self.normalize(de_centered_spectrum)


if __name__ == "__main__":
    bp = Bandpass(cv2.imread(r'/home/john/Desktop/ufo.cache-1.jpg', 0))
    cv2.imshow("bandpass_bitmap", bp.band_pass(outer_radius=50, inner_radius=10))
    cv2.waitKey()
