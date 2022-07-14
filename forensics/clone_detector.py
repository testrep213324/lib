import cv2
from scipy.spatial.distance import pdist

from forensics.helpers import image_resize


def seq_to_int(val):
    return tuple(map(int, val))


def key_points_matcher(bf, kp, des, ratio=0.6, distance=10):
    descriptor_matches = bf.knnMatch(des, des, k=10)
    points = []
    for d_matches in descriptor_matches:
        for match in range(1, len(d_matches) - 1):

            if d_matches[match].distance > d_matches[match + 1].distance * ratio:
                continue

            point_q = kp[d_matches[match].queryIdx].pt
            point_t = kp[d_matches[match].trainIdx].pt

            if pdist(X=(point_q, point_t), metric='euclidean') < distance:
                continue

            point_t = seq_to_int(point_t)
            point_q = seq_to_int(point_q)

            if (point_t, point_q) in points:
                continue

            points.append((point_q, point_t))

    return tuple(points)


def get_clones(cv_rgb_bitmap):
    local_cv_rgb_bitmap = image_resize(cv_rgb_bitmap.copy(), width=1000)

    sift = cv2.SIFT_create()
    bf = cv2.BFMatcher(cv2.NORM_L2)

    raw_key_points = sift.detect(local_cv_rgb_bitmap)
    key_points, descriptors = sift.compute(local_cv_rgb_bitmap, raw_key_points)

    key_point_pairs = key_points_matcher(bf, key_points, descriptors)

    for point_q, point_t, in key_point_pairs:
        cv2.circle(img=local_cv_rgb_bitmap, center=point_q, radius=3, color=(0, 255, 0), thickness=2)
        cv2.circle(img=local_cv_rgb_bitmap, center=point_t, radius=3, color=(0, 0, 255), thickness=2)
        cv2.line(img=local_cv_rgb_bitmap, pt1=point_q, pt2=point_t, color=(255, 0, 0))

    return local_cv_rgb_bitmap, len(key_point_pairs)


if __name__ == "__main__":
    img = cv2.imread(r'/home/john/Desktop/ufo.cache-1.jpg')
    clone_bitmap, features_number = get_clones(img)
    print("gg", clone_bitmap.shape)
    cv2.imshow("Result", clone_bitmap)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
