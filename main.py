import sys
import cv2
import numpy as np
from PyQt5 import uic
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QFileDialog, QAction, QInputDialog, QMessageBox
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt

class ImageProcessingWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('Image Processing')
        # 从文件中加载UI定义
        self.ui = uic.loadUi("main.ui")
        # 加载初始提醒图片
        pix = QPixmap(r'init.jpg')
        self.image = cv2.imread(r'init.jpg')
        self.ui.image_label.setPixmap(pix.scaled(self.ui.image_label.size(), Qt.KeepAspectRatio))

        self.ui.b_open_image.clicked.connect(self.open_image)
        self.ui.b_save_image.clicked.connect(self.save_image)
        self.ui.b_vertical_flip.clicked.connect(self.vertical_flip)
        self.ui.b_horizontal_flip.clicked.connect(self.horizontal_flip)
        self.ui.b_rotate_image.clicked.connect(self.rotate_image)
        self.ui.b_crop_image.clicked.connect(self.crop_image)
        self.ui.b_convert_to_gray.clicked.connect(self.convert_to_gray)
        self.ui.b_gaussian_blur.clicked.connect(self.apply_gaussian_blur)
        self.ui.b_sharpen.clicked.connect(self.sharpen)
        self.ui.b_linear_transform.clicked.connect(self.apply_linear_transform)

        # 设置默认卷积核参数
        self.ui.k1.setText('0')
        self.ui.k2.setText('0')
        self.ui.k3.setText('0')
        self.ui.k4.setText('0')
        self.ui.k5.setText('1')
        self.ui.k6.setText('0')
        self.ui.k7.setText('0')
        self.ui.k8.setText('0')
        self.ui.k9.setText('0')
        self.ui.b_start_conv.clicked.connect(self.start_conv)
        self.ui.b_x_sobel.clicked.connect(self.set_x_sobel)
        self.ui.b_y_sobel.clicked.connect(self.set_y_sobel)


    def show_cv_image(self, image):
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        height, width, channel = image_rgb.shape
        qimage = QImage(image_rgb.data, width, height, width * channel, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qimage)

        self.ui.image_label.setPixmap(pixmap.scaled(self.ui.image_label.size(), Qt.KeepAspectRatio))

    def open_image(self):
        file_dialog = QFileDialog()
        file_path = file_dialog.getOpenFileName(self, 'Open Image')[0]
        if file_path:
            image = cv2.imread(file_path)
            self.image = image
            self.show_cv_image(image)

    def save_image(self):
        if self.image is not None:
            file_dialog = QFileDialog()
            file_path = file_dialog.getSaveFileName(self, 'Save Image')[0]
            print(file_path)
            if file_path:
                cv2.imwrite(file_path, self.image)

    def vertical_flip(self):
        if self.image is not None:
            flipped_image = cv2.flip(self.image, 0)
            self.image = flipped_image
            self.show_cv_image(self.image)

    def horizontal_flip(self):
        if self.image is not None:
            flipped_image = cv2.flip(self.image, 1)
            self.image = flipped_image
            self.show_cv_image(self.image)

    def rotate_image(self):
        if self.image is not None:
            angle, ok = QInputDialog.getInt(self, 'Rotate Image', '输入要旋转的角度（顺时针为正）:')

            if ok:
                rows, cols, _ = self.image.shape
                rotation_matrix = cv2.getRotationMatrix2D((cols / 2, rows / 2), -angle, 1)
                rotated_image = cv2.warpAffine(self.image, rotation_matrix, (cols, rows))
                self.image = rotated_image
                self.show_cv_image(self.image)

    #图像裁剪
    def crop_image(self):
        if self.image is not None:
            roi = cv2.selectROI('Please choose the scope, then press enter!', self.image, fromCenter=False, showCrosshair=True)
            if roi != (0, 0, 0, 0):
                cropped_image = self.image[int(roi[1]):int(roi[1]+roi[3]), int(roi[0]):int(roi[0]+roi[2])]
                self.image = cropped_image
                self.show_cv_image(self.image)
                cv2.waitKey(0)
                cv2.destroyAllWindows()


    #色彩图转灰度图
    def convert_to_gray(self):
        if self.image is not None:
            gray_image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
            self.image = gray_image
            self.show_cv_image(self.image)
            # cv2.imshow('Gray Image', gray_image)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()

    #高斯滤波
    def apply_gaussian_blur(self):
        if self.image is not None:
            ksize, ok = QInputDialog.getInt(self, 'Gaussian Blur', '请输入核大小 (奇数):')
            if ok:
                # 判断核大小是正奇数
                if ksize < 1 or ksize % 2 == 0:
                    QMessageBox.warning(self, "警告", "请输入正奇数！如1、3、5等", QMessageBox.Cancel)
                else:
                    blurred_image = cv2.GaussianBlur(self.image, (ksize, ksize), 0)
                    self.show_cv_image(blurred_image)
    # 图像锐化
    def sharpen(self):
        sharpen_k = np.array([[0, -1, 0],
                              [-1, 5, -1],
                              [0, -1, 0]], dtype=np.float32)
        s_image = cv2.filter2D(self.image, cv2.CV_32F, sharpen_k)
        self.image = cv2.convertScaleAbs(s_image)
        self.show_cv_image(self.image)
    #直方图均值化
    def apply_histogram_equalization(self):
        if self.image is not None:
            gray_image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
            equalized_image = cv2.equalizeHist(gray_image)
            colored_equalized_image = cv2.cvtColor(equalized_image, cv2.COLOR_GRAY2BGR)
            self.show_cv_image(colored_equalized_image)

    #线性变化
    def apply_linear_transform(self):
        if self.image is not None:
            alpha, ok = QInputDialog.getDouble(self, 'Linear Transform', 'Enter alpha value:')
            if ok:
                beta, ok = QInputDialog.getDouble(self, 'Linear Transform', 'Enter beta value:')
                if ok:
                    transformed_image = cv2.convertScaleAbs(self.image, alpha=alpha, beta=beta)
                    self.show_cv_image(transformed_image)

    #伽马变换
    def apply_gamma_transform(self):
        if self.image is not None:
            gamma, ok = QInputDialog.getDouble(self, 'Gamma Transform', 'Enter gamma value:')
            if ok:
                gamma_corrected = np.power(self.image / 255.0, gamma)
                gamma_corrected = (gamma_corrected * 255).astype(np.uint8)
                self.show_cv_image(gamma_corrected)

    def conv(self, conv_k, stride):
        # conv_k = np.array([[0, -1, 0],
        #                       [-1, 5, -1],
        #                       [0, -1, 0]], dtype=np.float32)
        s_image = cv2.filter2D(self.image, cv2.CV_32F, conv_k)
        self.image = cv2.convertScaleAbs(s_image)
        self.show_cv_image(self.image)

    def start_conv(self):
        try:
            k1 = float(self.ui.k1.text())
            k2 = float(self.ui.k2.text())
            k3 = float(self.ui.k3.text())
            k4 = float(self.ui.k4.text())
            k5 = float(self.ui.k5.text())
            k6 = float(self.ui.k6.text())
            k7 = float(self.ui.k7.text())
            k8 = float(self.ui.k8.text())
            k9 = float(self.ui.k9.text())
            conv_k = np.array([[k1, k2, k3],
                               [k4, k5, k6],
                               [k7, k8, k9]], dtype=np.float32)
            self.conv(conv_k, 1)
        except ValueError:
            QMessageBox.warning(self, "警告", "卷积核参数请输入阿拉伯数字！", QMessageBox.Cancel)

    def set_x_sobel(self):
        self.ui.k1.setText('-1')
        self.ui.k2.setText('0')
        self.ui.k3.setText('1')
        self.ui.k4.setText('-2')
        self.ui.k5.setText('0')
        self.ui.k6.setText('2')
        self.ui.k7.setText('-1')
        self.ui.k8.setText('0')
        self.ui.k9.setText('1')

    def set_y_sobel(self):
        self.ui.k1.setText('1')
        self.ui.k2.setText('2')
        self.ui.k3.setText('1')
        self.ui.k4.setText('0')
        self.ui.k5.setText('0')
        self.ui.k6.setText('0')
        self.ui.k7.setText('-1')
        self.ui.k8.setText('-2')
        self.ui.k9.setText('-1')



if __name__ == '__main__':
    app = QApplication(sys.argv)

    main_window = ImageProcessingWindow()
    main_window.ui.show()

    sys.exit(app.exec_())