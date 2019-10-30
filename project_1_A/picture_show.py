import cv2


#将照品显示出来
def show(img, name):
    cv2.imshow(name, img)
    key = cv2.waitKey()
    if key == 27:
        cv2.destroyAllWindows()


# 原图显示
def Or_show(path):
    g_pic = Pic(path)
    img = g_pic.pic_c()
    show(img, 'Original')


#图像读取函数
class Pic(object):
    def __init__(self, path):
        self.path = path

    def pic_c(self):
        img_c = cv2.imread(self.path, 1)
        return img_c

    def pic_g(self):
        img_g = cv2.imread(self.path, 0)
        return img_g

    def pic_RGB(self):
        B, G, R = cv2.split(self.pic_c())
        return B, G, R