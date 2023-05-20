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
        self.ui.image_label.setPixmap(pix.scaled(self.ui.image_label.size(), Qt.KeepAspectRatio))

        self.ui.b_open_image.clicked.connect(self.open_image)
        self.ui.b_save_image.clicked.connect(self.save_image)

    def open_image(self):
        file_dialog = QFileDialog()
        file_path = file_dialog.getOpenFileName(self, 'Open Image')[0]
        if file_path:
            image = cv2.imread(file_path)
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            height, width, channel = image_rgb.shape
            print(height+width+channel)
            qimage = QImage(image_rgb.data, width, height, width * channel, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(qimage)

            self.ui.image_label.setPixmap(pixmap.scaled(self.ui.image_label.size(), Qt.KeepAspectRatio))

            self.image = image
    def save_image(self):
        if self.image is not None:
            file_dialog = QFileDialog()
            file_path = file_dialog.getSaveFileName(self, 'Save Image')[0]
            print(file_path)
            if file_path:
                cv2.imwrite(file_path, self.image)

# class ImageProcessingApp(QWidget):
#     def __init__(self):
#         super().__init__()

if __name__ == '__main__':
    app = QApplication(sys.argv)

    main_window = ImageProcessingWindow()
    main_window.ui.show()

    # image_app = ImageProcessingApp()
    # image_app.show()

    sys.exit(app.exec_())