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
        self.ui.pushButton_12.clicked.connect(self.open_image)

    def open_image(self):
        file_dialog = QFileDialog()
        file_path = file_dialog.getOpenFileName(self, 'Open Image')[0]
        if file_path:
            image = cv2.imread(file_path)
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            height, width, channel = image_rgb.shape
            qimage = QImage(image_rgb.data, width, height, width * channel, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(qimage)
            print(self.ui)
            print(self.ui.label_5)
            self.ui.label_5.setPixmap(pixmap.scaled(self.ui.label_5.size(), Qt.KeepAspectRatio))

            self.image = image

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