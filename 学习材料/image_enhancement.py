import cv2
import numpy as np
# import matplotlib.pyplot as plt


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

# 线性变换
def linear_trans(img):
    # 图像灰度转换
    grayImage = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # 获取图像高度和宽度
    height = grayImage.shape[0]
    width = grayImage.shape[1]
    # 创建一幅图像
    result = np.zeros((height, width), np.uint8)
    # 图像对比度增强变换 DB=DA*3
    for i in range(height):
        for j in range(width):
            if (int(grayImage[i, j] * 3) > 255):
                gray = 255
            else:
                gray = int(grayImage[i, j] * 3)
            result[i, j] = np.uint8(gray)
    # 显示图像
    cv2.imshow("Gray Image", grayImage)
    cv2.imshow("Result", result)
    cv2.waitKey(delay=0)


# 下面用于测试函数效果
# test_img = blur(img)
#test_img = warp(img,250,100)
#cv2.imshow("image", test_img)
test_linear = linear_trans(img)
cv2.waitKey(0)
cv2.destroyAllWindows()