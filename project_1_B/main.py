import cv2

import keypointmatch
from out_put import log


def mian():
    imname_1 = 'IMG_2411.jpg'
    imname_2 = 'IMG_2412.jpg'
    image1 = cv2.imread(imname_1)
    image2 = cv2.imread(imname_2)
    stitch_match = keypointmatch.FindKeyPointsAndMatch()
    kp1, kp2 = stitch_match.get_key_points(image1, image2)
    homo_matrix = stitch_match.match(kp1, kp2)
    # stitch_merge = keypointmatch.PasteTwoImages()
    # merge_image = stitch_merge(image1, image2, homo_matrix)
    # cv2.imshow("output", merge_image)
    # if cv2.waitKey() == 27:
    #     cv2.destroyAllWindows()
    # cv2.imwrite(" lenna_merge.jpg ", merge_image)
    # log("\n=============>Output saved!")


if __name__ == '__main__':
    mian()
    # pass
