import cv2
import numpy as np
# from matplotlib import pyplot as plt
from sklearn.decomposition import PCA


def pca_helper(img, comp_a, comp_b):
    # https://onlinemschool.com/math/assistance/cartesian_coordinate/p_line/
    start_xyz = np.ones_like(img) * comp_a
    stop_xyz = np.ones_like(img) * (comp_b - comp_a)

    mms = np.abs(np.cross(start_xyz - img, stop_xyz))
    a = np.abs(mms).sum(axis=2)
    b = np.abs(stop_xyz).sum(axis=2)
    rez = np.abs(np.abs(a) / np.abs(b))
    return np.array(rez * (255 / rez.max()), dtype=np.uint8)


# def plotting(img, decim_step=10):
#     img_arr = img.transpose(0, 1, 2).reshape(-1, 3)
#     fig = plt.figure(figsize=(7, 5))
#     ax = fig.add_subplot(111, projection='3d')
#     ax.scatter((img_arr[:, 0])[0::decim_step], (img_arr[:, 1])[0::decim_step], (img_arr[:, 2])[0::decim_step])
#     ax.set_xlabel("Blue")
#     ax.set_ylabel("Green")
#     ax.set_zlabel("Red", rotation=90)
#     plt.tight_layout()
#     plt.show()


def pca_analysis(img, component_choice=123):
    img_arr = img.transpose(0, 1, 2).reshape(-1, 3)
    pca = PCA(n_components=3)
    pca.fit(img_arr)
    centroid = np.mean(img_arr, 0)

    comp00, comp10, comp20 = centroid + pca.components_ * np.array([-255])
    comp01, comp11, comp21 = centroid + pca.components_ * np.array([254])

    component_choices = {1: pca_helper(img, comp_a=comp00, comp_b=comp01),
                         2: pca_helper(img, comp_a=comp10, comp_b=comp11),
                         3: pca_helper(img, comp_a=comp20, comp_b=comp21),
                         123: np.stack([pca_helper(img, comp_a=comp00, comp_b=comp01),
                                        pca_helper(img, comp_a=comp10, comp_b=comp11),
                                        pca_helper(img, comp_a=comp20, comp_b=comp21)], axis=-1)}

    return component_choices[component_choice]


if __name__ == "__main__":
    img = cv2.imread('/home/john/Desktop/ufo.cache-1.jpg')
    img = cv2.resize(img, (640, 480))
    pca_bitmap = pca_analysis(img, component_choice=123)
    cv2.imshow('pca_bitmap', pca_bitmap)
    # plotting(img, decim_step=10)
    cv2.waitKey(0)
