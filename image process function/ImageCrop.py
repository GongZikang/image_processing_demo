import sys
import cv2
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QVBoxLayout, QPushButton, QFileDialog

#图像裁剪功能：上传图片，选择好裁剪区域后，点击enter键显示裁剪后的图像并进行保存；

class ImageCropper(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('Image Cropper')
        self.layout = QVBoxLayout()

        self.image_label = QLabel()
        self.layout.addWidget(self.image_label)

        self.crop_button = QPushButton('Crop Image')
        self.crop_button.clicked.connect(self.crop_image)
        self.layout.addWidget(self.crop_button)

        self.setLayout(self.layout)
        self.image = None

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

    def crop_image(self):
        if self.image is not None:
            roi = cv2.selectROI('Select ROI', self.image, fromCenter=False, showCrosshair=True)
            if roi != (0, 0, 0, 0):
                cropped_image = self.image[int(roi[1]):int(roi[1]+roi[3]), int(roi[0]):int(roi[0]+roi[2])]
                cv2.imshow('Cropped Image', cropped_image)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
                save_dialog = QFileDialog()
                save_path, _ = save_dialog.getSaveFileName(self, 'Save Image', '', 'Image Files (*.png *.jpg *.jpeg)')
                if save_path:
                    cv2.imwrite(save_path, cropped_image)


if __name__ == '__main__':
    app = QApplication(sys.argv)

    cropper = ImageCropper()
    cropper.show()

    file_dialog = QFileDialog()
    file_path, _ = file_dialog.getOpenFileName(cropper, 'Open Image', '', 'Image Files (*.png *.jpg *.jpeg)')
    if file_path:
        cropper.load_image(file_path)

    sys.exit(app.exec_())
