import cv2
import numpy as np

from out_put import log

class FindKeyPointsAndMatch(object):
    def __init__(self):
        self.sift = cv2.xfeatures2d.SIFT_create()
        self.brute = cv2.BFMatcher()

    def get_key_points(self, img1, img2):
        g_img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        g_img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
        kp1, kp2 = {}, {}

        #关键点检测
        kp1["kp"], kp1["des"] = self.sift.detectAndCompute(g_img1, None)
        kp2["kp"], kp2["des"] = self.sift.detectAndCompute(g_img2, None)
        return kp1, kp2

    #关键点匹配
    def match(self, kp1,kp2):
        matches = self.brute.knnMatch(kp1["des"], kp2["des"], k = 2)
        good_matches = []

        for i ,(m,n) in enumerate(matches):
            if m.distance < 0.7 * n.distance:
                good_matches.append((m.trainIdx, m.queryIdx))

        if len(good_matches) > 4:
            key_points1 = kp1["kp"]
            key_points2 = kp2["kp"]

            matched_kp1 = np.float32(
                [key_points1[i].pt for (_, i) in good_matches]
            )

            matched_kp2 = np.float32(
                [key_points2[i].pt for (i, _) in good_matches]
            )

            #求解特征矩阵

            homo_matrix, mask = cv2.findHomography(matched_kp1, matched_kp2, cv2.RANSAC, 4)
            log('homo_matrix', homo_matrix)
            return homo_matrix, mask
        else:
            return None

class picture_operation(object):
    def __init__(self):
        pass

    def find_corner(self, mask):
        high, wide = img1.shape

        corner_point = np.float32([[0, 0], [0, high - 1], [wide - 1, high - 1], [wide - 1, 0]]).reshape(-1, 1, 2)
        cg_corner_point = cv2.perspectiveTransform(corner_point, mask)

# class PasteTwoImages(object):
#     def __init__(self):
#         pass
#
#     def __call__(self, img1, img2, homo_matrix):
#         h1, w1 = img1.shape[0], img1.shape[1]
#         h2, w2 = img2.shape[0], img2.shape[1]
#         rect1 = np.array([[0,0],[0,h1],[w1,h1],[w1,0]], dtype=np.float32).reshape(4,1,2)
#         rect2 = np.array([[0,0], [0,h2], [w2,h2], [w2, 0]], dtype=np.float32).reshape(4,1,2)
#         trans_rect1 = cv2.perspectiveTransform(rect1, homo_matrix)
#         total_rect = np.concatenate((rect2, trans_rect1),axis=0)
#         min_x, min_y = np.int32(total_rect.min(axis=0).ravel())
#         max_x, max_y = np.int32(total_rect.max(axis=0).ravel())
#         shift_to_zero_matrix = np.array([[1,0,-min_x], [0, 1, -min_y],[0,0,1]])
#         trans_img1 = cv2.warpPerspective(img1, shift_to_zero_matrix.dot(homo_matrix), (max_x - min_x, max_y - min_y))
#         trans_img1[-min_y:h2 - min_y, -min_x:w2 - min_x] = img2
#         return trans_img1

