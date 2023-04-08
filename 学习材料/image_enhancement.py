import cv2


# 读取图片
img = cv2.imread("./test.jpg")
# 缩放图片
scale = 400/img.shape[0]
img = cv2.resize(img, dsize=None, fx=scale, fy=scale)
# 通道分离
bImg, gImg, rImg = cv2.split(img)

cv2.imwrite("r.jpg", rImg)
cv2.imwrite("b.jpg", bImg)
cv2.imwrite("g.jpg", gImg)

print(img.shape)