import cv2
import numpy as np


class DCT:
    def __init__(self, bgr_bitmap):
        self.bgr_bitmap = bgr_bitmap
        self.dct_hist = None

        self.hist_x = None
        self.hist_y = None
        self.hist_y_average = None

        self.changes = None
        self.calc_energy = None
        self.real_energy = None

        self.get_hist()
        self.norm_hist()
        self.avg_hist()
        self.calc_stat()

    def get_hist(self, comp=(2, 2)):
        # some images are not divisible by 8, they are cut off
        h, w = self.bgr_bitmap.shape[:2]
        h = int((h // 8) * 8)
        w = int((w // 8) * 8)
        bgr_bitmap = self.bgr_bitmap[0:h, 0:w]

        h, w = bgr_bitmap.shape[:2]
        y = cv2.cvtColor(bgr_bitmap, cv2.COLOR_BGR2YCR_CB)[:, :, 0]
        y = y.reshape(h // 8, 8, -1, 8).swapaxes(1, 2).reshape(-1, 8, 8)
        q_dct = []
        for i in range(0, y.shape[0]):
            q_dct.append(cv2.dct(np.float32(y[i])))
        q_dct = np.asarray(q_dct, dtype=np.float32)
        temp = q_dct - np.mean(q_dct, axis=0)
        q_dct = np.rint(temp).astype(np.int32)
        self.dct_hist = q_dct[:, comp[0], comp[1]]

    @staticmethod
    def deviation(spis):
        spis = list(spis)
        cnt = 0
        for i in range(len(spis) - 1):
            if spis[i] != spis[i + 1]:
                cnt += 1
        return cnt

    @staticmethod
    def mov_filter(orig_list, window=3, step=1, f_type="mean"):
        func = np.mean
        if f_type == "median":
            func = np.median
        elif f_type == "min":
            func = np.min

        if window % 2 != 0:
            first = window // 2
            back = first
        else:
            first = (window // 2) - 1
            back = first + 1

        orig_list = np.concatenate((orig_list, np.ones(back) * orig_list[-1]))

        orig_list = np.concatenate((np.ones(first) * orig_list[0], orig_list))

        rez = []
        i = 0
        while i < len(orig_list) - (window - 1):
            rez.append(func(orig_list[i:][:window]))
            i += step

        return np.maximum(rez, np.roll(rez, -1))

    def norm_hist(self):
        hist_y, hist_x = np.histogram(self.dct_hist, max(self.dct_hist) - min(self.dct_hist))
        self.hist_x = hist_x[:-1]
        # histogram normalization (LOG 0-1)
        hist_y = np.where(hist_y < 1, 1, hist_y)
        hist_y = np.log(hist_y)
        self.hist_y = hist_y / max(hist_y)

    def avg_hist(self):
        hist_y_min = self.mov_filter(self.hist_y, window=3, step=1, f_type="min")
        self.hist_y_average = self.mov_filter(hist_y_min, window=4, step=1, f_type="mean")

    def calc_stat(self, x_anls_lim_0=-40, x_anls_lim_1=40):
        point = self.hist_x.min()
        if point < x_anls_lim_0:
            point = x_anls_lim_0
        start_ind = list(self.hist_x).index(point)
        cut_length = abs(x_anls_lim_0) + x_anls_lim_1 + 1

        # converts the histogram into sequences 0-1 and count the number of state chang
        threshed = self.hist_y[start_ind:][: cut_length]
        threshed = np.where(threshed == 0, 0, 1)
        self.changes = self.deviation(threshed)

        # the difference between the original and truncated histogram, and discarding the negative component
        self.calc_energy = float(sum(self.hist_y_average[start_ind:][:cut_length]))
        self.real_energy = float(sum(self.hist_y[start_ind:][:cut_length]))

    @staticmethod
    def draw(dct_hist):
        from matplotlib import pyplot as plt
        fig, axs = plt.subplots(1)
        axs.cla()
        x_anls_lim_0, x_anls_lim_1 = -40, 40
        axs.bar(dct_hist.hist_x - 1.0 / 2, dct_hist.hist_y, width=1.0)
        axs.plot(dct_hist.hist_x, dct_hist.hist_y_average, color='r')
        axs.set_ylabel("histogram", fontsize=10)
        axs.axvline(x_anls_lim_0, color='black', linewidth=1)
        axs.axvline(x_anls_lim_1, color='black', linewidth=1)
        axs.set_xlim(-100, 100)

        title = f"sum red:{np.round(dct_hist.calc_energy, 2)} " \
                f" sum blue:{np.round(dct_hist.real_energy, 2)}" \
                f" drops: {np.round(dct_hist.changes, 2)}"

        axs.set_title(title)
        plt.subplots_adjust(hspace=0.35)
        plt.show()

    def jsonify(self):
        x = self.hist_x.astype(int).tolist()
        y = self.hist_y.round(4).tolist()
        y_avg = self.hist_y_average.round(4).tolist()

        e_calc = float(self.calc_energy)
        e_real = float(self.real_energy)
        e_drops = float(self.changes)
        return {"x": x, "y": y, "y_avg": y_avg, "e_calc": e_calc, "e_real": e_real, "e_drops": e_drops}


if __name__ == "__main__":
    cv_bgr_bitmap = cv2.imread('/home/john/Desktop/ufo.cache-1.jpg')
    local_cv_bgr_bitmap = cv_bgr_bitmap.copy()
    dct_h = DCT(local_cv_bgr_bitmap)
    dct_h.jsonify()
