import sys
import cv2
from PyQt5 import uic
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QFileDialog, QAction, QInputDialog
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




if __name__ == '__main__':
    app = QApplication(sys.argv)

    main_window = ImageProcessingWindow()
    main_window.ui.show()


    sys.exit(app.exec_())