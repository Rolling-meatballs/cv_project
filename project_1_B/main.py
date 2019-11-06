import cv2

import keypointmatch
from picture_show import show
from out_put import log


def mian():
    imname_1 = 'kindle.jpg'
    imname_2 = 'IMG_2460.jpg'
    image1 = cv2.imread(imname_1)
    image2 = cv2.imread(imname_2)
    stitch_match = keypointmatch.FindKeyPointsAndMatch()
    kp1, kp2 = stitch_match.get_key_points(image1, image2)
    # log('kp1', kp1)
    # log('kp2', kp2)
    homo_matrix, mask, good_matches = stitch_match.match(kp1, kp2)

    pic_op = keypointmatch.picture_operation(image1, image2)
    frame = pic_op.find_frame(homo_matrix)
    cv2.imwrite('frame.jpg', frame)
    img = pic_op.picture_show(kp1, kp2, mask, good_matches, frame)

    cv2.imwrite('end{}'.format(image1), img)
    show(img, 'edn{}'.format(image1))

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
