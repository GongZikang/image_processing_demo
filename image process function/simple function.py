import sys
import cv2
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QVBoxLayout, QPushButton, QFileDialog, QSlider, \
    QMainWindow, QAction, QInputDialog
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt

#定义一些简单功能：垂直翻转、水平翻转、图片旋转、色彩图转灰度图

class ImageProcessingWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('Image Processing')
        self.image_label = QLabel()
        self.setCentralWidget(self.image_label)
        self.create_menu()

    def create_menu(self):
        open_action = QAction('Open Image', self)
        open_action.setShortcut('Ctrl+O')
        open_action.triggered.connect(self.open_image)

        save_action = QAction('Save Image', self)
        save_action.setShortcut('Ctrl+S')
        save_action.triggered.connect(self.save_image)

        self.menuBar().addMenu('File').addAction(open_action)
        self.menuBar().addMenu('File').addAction(save_action)

    def open_image(self):
        file_dialog = QFileDialog()
        file_path = file_dialog.getOpenFileName(self, 'Open Image')[0]
        if file_path:
            image = cv2.imread(file_path)
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            height, width, channel = image_rgb.shape
            qimage = QImage(image_rgb.data, width, height, width * channel, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(qimage)
            self.image_label.setPixmap(pixmap.scaled(self.image_label.size(), Qt.KeepAspectRatio))

            self.image = image

    def save_image(self):
        file_dialog = QFileDialog()
        file_path = file_dialog.getSaveFileName(self, 'Save Image')[0]
        if file_path:
            cv2.imwrite(file_path, self.image)


class ImageProcessingApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('Image Processing')
        self.layout = QVBoxLayout()

        self.open_button = QPushButton('Open Image')
        self.open_button.clicked.connect(self.open_image)

        self.horizontal_flip_button = QPushButton('Horizontal Flip')
        self.horizontal_flip_button.clicked.connect(self.horizontal_flip)

        self.vertical_flip_button = QPushButton('Vertical Flip')
        self.vertical_flip_button.clicked.connect(self.vertical_flip)

        self.rotate_button = QPushButton('Rotate Image')
        self.rotate_button.clicked.connect(self.rotate_image)

        self.convert_to_gray_button = QPushButton('Convert to Gray')
        self.convert_to_gray_button.clicked.connect(self.convert_to_gray)

        self.layout.addWidget(self.open_button)
        self.layout.addWidget(self.horizontal_flip_button)
        self.layout.addWidget(self.vertical_flip_button)
        self.layout.addWidget(self.rotate_button)
        self.layout.addWidget(self.convert_to_gray_button)


        self.setLayout(self.layout)

        self.image = None

    def open_image(self):
        file_dialog = QFileDialog()
        file_path = file_dialog.getOpenFileName(self, 'Open Image')[0]
        if file_path:
            self.image = cv2.imread(file_path)

    #垂直翻转
    def horizontal_flip(self):
        if self.image is not None:
            flipped_image = cv2.flip(self.image, 1)
            cv2.imshow('Flipped Image', flipped_image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

    #水平翻转
    def vertical_flip(self):
        if self.image is not None:
            flipped_image = cv2.flip(self.image, 0)
            cv2.imshow('Flipped Image', flipped_image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

    #图片旋转
    def rotate_image(self):
        if self.image is not None:
            angle, ok = QInputDialog.getDouble(self, 'Rotate Image', 'Enter angle (in degrees):')
            if ok:
                rows, cols, _ = self.image.shape
                rotation_matrix = cv2.getRotationMatrix2D((cols / 2, rows / 2), angle, 1)
                rotated_image = cv2.warpAffine(self.image, rotation_matrix, (cols, rows))
                cv2.imshow('Rotated Image', rotated_image)
                cv2.waitKey(0)
                cv2.destroyAllWindows()

    #色彩图转灰度图
    def convert_to_gray(self):
        if self.image is not None:
            gray_image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
            cv2.imshow('Gray Image', gray_image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()




if __name__ == '__main__':
    app = QApplication(sys.argv)

    main_window = ImageProcessingWindow()
    main_window.show()

    image_app = ImageProcessingApp()
    image_app.show()

    sys.exit(app.exec_())
