import sys
import cv2
import numpy as np
from PyQt5 import uic
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QVBoxLayout, QPushButton, QFileDialog, QInputDialog

#此模块实现功能有：
#1、高斯滤波：用户可自定义kenal size
#2、直方图均值化
#3、线性变化：用户可自定义y=ax+b中的a、b的值
#4、伽马变换：用户可自定义gamma值，值大于1时图像亮度增强，小于1时亮度减小
class ImageProcessor(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('Image Processor')
        self.layout = QVBoxLayout()

        self.image_label = QLabel()
        self.layout.addWidget(self.image_label)

        self.processed_image_label = QLabel()
        self.layout.addWidget(self.processed_image_label)

        self.gaussian_blur_button = QPushButton('Gaussian Blur')
        self.gaussian_blur_button.clicked.connect(self.apply_gaussian_blur)
        self.layout.addWidget(self.gaussian_blur_button)

        self.histogram_equalization_button = QPushButton('Histogram Equalization')
        self.histogram_equalization_button.clicked.connect(self.apply_histogram_equalization)
        self.layout.addWidget(self.histogram_equalization_button)

        self.linear_transform_button = QPushButton('Linear Transform')
        self.linear_transform_button.clicked.connect(self.apply_linear_transform)
        self.layout.addWidget(self.linear_transform_button)

        self.gamma_transform_button = QPushButton('Gamma Transform')
        self.gamma_transform_button.clicked.connect(self.apply_gamma_transform)
        self.layout.addWidget(self.gamma_transform_button)

        self.save_button = QPushButton('Save Image')
        self.save_button.clicked.connect(self.save_image)
        self.layout.addWidget(self.save_button)

        self.setLayout(self.layout)
        self.image = None
        self.processed_image = None

    def load_image(self, file_path):
        image = cv2.imread(file_path)
        self.display_image(image)

    def display_image(self, image):
        if image is not None:
            self.image = image
            height, width, channel = image.shape
            qimage = QImage(image.data, width, height, width * channel, QImage.Format_BGR888)
            pixmap = QPixmap.fromImage(qimage)
            self.image_label.setPixmap(pixmap.scaled(self.image_label.size(), Qt.KeepAspectRatio))
        else:
            self.image_label.clear()

    def display_processed_image(self, image):
        if image is not None:
            self.processed_image = image
            height, width, channel = image.shape
            qimage = QImage(image.data, width, height, width * channel, QImage.Format_BGR888)
            pixmap = QPixmap.fromImage(qimage)
            self.processed_image_label.setPixmap(pixmap.scaled(self.processed_image_label.size(), Qt.KeepAspectRatio))
        else:
            self.processed_image_label.clear()

    #高斯滤波
    def apply_gaussian_blur(self):
        if self.image is not None:
            ksize, ok = QInputDialog.getInt(self, 'Gaussian Blur', 'Enter kernel size (odd number):')
            if ok:
                blurred_image = cv2.GaussianBlur(self.image, (ksize, ksize), 0)
                self.display_processed_image(blurred_image)

    #直方图均值化
    def apply_histogram_equalization(self):
        if self.image is not None:
            gray_image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
            equalized_image = cv2.equalizeHist(gray_image)
            colored_equalized_image = cv2.cvtColor(equalized_image, cv2.COLOR_GRAY2BGR)
            self.display_processed_image(colored_equalized_image)

    #线性变化
    def apply_linear_transform(self):
        if self.image is not None:
            alpha, ok = QInputDialog.getDouble(self, 'Linear Transform', 'Enter alpha value:')
            if ok:
                beta, ok = QInputDialog.getDouble(self, 'Linear Transform', 'Enter beta value:')
                if ok:
                    transformed_image = cv2.convertScaleAbs(self.image, alpha=alpha, beta=beta)
                    self.display_processed_image(transformed_image)

    #伽马变换
    def apply_gamma_transform(self):
        if self.image is not None:
            gamma, ok = QInputDialog.getDouble(self, 'Gamma Transform', 'Enter gamma value:')
            if ok:
                gamma_corrected = np.power(self.image / 255.0, gamma)
                gamma_corrected = (gamma_corrected * 255).astype(np.uint8)
                self.display_processed_image(gamma_corrected)

    def save_image(self):
        if self.processed_image is not None:
            file_path, _ = QFileDialog.getSaveFileName(self, 'Save Image', '', 'Image Files (*.png *.jpg *.jpeg)')
            if file_path:
                cv2.imwrite(file_path, self.processed_image)


if __name__ == '__main__':
    app = QApplication(sys.argv)

    processor = ImageProcessor()
    processor.show()

    file_dialog = QFileDialog()
    file_path, _ = file_dialog.getOpenFileName(processor, 'Open Image', '', 'Image Files (*.png *.jpg *.jpeg)')
    if file_path:
        processor.load_image(file_path)

    sys.exit(app.exec_())
