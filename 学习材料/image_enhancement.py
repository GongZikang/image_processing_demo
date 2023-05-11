import cv2
import numpy as np
import matplotlib.pyplot as plt


# 读取图片
img = cv2.imread("./test.jpg")
# 缩放图片
scale = 400/img.shape[0]
img = cv2.resize(img, dsize=None, fx=scale, fy=scale)
# # 通道分离
# bImg, gImg, rImg = cv2.split(img)
#
# cv2.imwrite("r.jpg", rImg)
# cv2.imwrite("b.jpg", bImg)
# cv2.imwrite("g.jpg", gImg)

print(img.shape)


# 图像平滑，使用opencv的cv.blur
def blur(img):
    blur = cv2.blur(img,(3,3))
    return blur


# 平移变换，使用opencv的 cv.warpAffine,第一个参数是向右平移量，第二个参数是向下平移量
def warp(img,a,b):
    rows, cols = img.shape[:2]
    M = np.float32([[1, 0, a], [0, 1, b]])

    img = cv2.warpAffine(img, M, (cols, rows))
    return img



# 下面用于测试函数效果
# test_img = blur(img)
test_img = warp(img,250,100)
cv2.imshow("image", test_img)
cv2.waitKey(0)
cv2.destroyAllWindows()