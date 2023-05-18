import sys
import cv2
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QVBoxLayout, QPushButton, QFileDialog, QSlider, \
    QMainWindow, QAction, QInputDialog
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt


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

        self.crop_button = QPushButton('Crop Image')
        self.crop_button.clicked.connect(self.crop_image)

        self.horizontal_flip_button = QPushButton('Horizontal Flip')
        self.horizontal_flip_button.clicked.connect(self.horizontal_flip)

        self.vertical_flip_button = QPushButton('Vertical Flip')
        self.vertical_flip_button.clicked.connect(self.vertical_flip)

        self.rotate_button = QPushButton('Rotate Image')
        self.rotate_button.clicked.connect(self.rotate_image)

        self.add_noise_button = QPushButton('Add Gaussian Noise')
        self.add_noise_button.clicked.connect(self.add_noise)

        self.convert_to_gray_button = QPushButton('Convert to Gray')
        self.convert_to_gray_button.clicked.connect(self.convert_to_gray)

        self.histogram_equalization_button = QPushButton('Histogram Equalization')
        self.histogram_equalization_button.clicked.connect(self.histogram_equalization)

        self.linear_transform_button = QPushButton('Linear Transform')
        self.linear_transform_button.clicked.connect(self.linear_transform)

        self.gamma_transform_button = QPushButton('Gamma Transform')
        self.gamma_transform_button.clicked.connect(self.gamma_transform)

        self.layout.addWidget(self.open_button)
        self.layout.addWidget(self.crop_button)
        self.layout.addWidget(self.horizontal_flip_button)
        self.layout.addWidget(self.vertical_flip_button)
        self.layout.addWidget(self.rotate_button)
        self.layout.addWidget(self.add_noise_button)
        self.layout.addWidget(self.convert_to_gray_button)
        self.layout.addWidget(self.histogram_equalization_button)
        self.layout.addWidget(self.linear_transform_button)
        self.layout.addWidget(self.gamma_transform_button)

        self.setLayout(self.layout)

        self.image = None

    def open_image(self):
        file_dialog = QFileDialog()
        file_path = file_dialog.getOpenFileName(self, 'Open Image')[0]
        if file_path:
            self.image = cv2.imread(file_path)

    def crop_image(self):
        if self.image is not None:
            height, width, _ = self.image.shape
            crop_width, crop_height, ok = self.get_crop_size(width, height)
            if ok:
                x = (width - crop_width) // 2
                y = (height - crop_height) // 2
                cropped_image = self.image[y:y+crop_height, x:x+crop_width]
                cv2.imshow('Cropped Image', cropped_image)
                cv2.waitKey(0)
                cv2.destroyAllWindows()

    def get_crop_size(self, max_width, max_height):
        width, ok1 = QInputDialog.getInt(self, 'Crop Image', 'Enter width:')
        height, ok2 = QInputDialog.getInt(self, 'Crop Image', 'Enter height:')
        return width, height, ok1 and ok2 and 0 < width <= max_width and 0 < height <= max_height

    def horizontal_flip(self):
        if self.image is not None:
            flipped_image = cv2.flip(self.image, 1)
            cv2.imshow('Flipped Image', flipped_image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

    def vertical_flip(self):
        if self.image is not None:
            flipped_image = cv2.flip(self.image, 0)
            cv2.imshow('Flipped Image', flipped_image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

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

    def add_noise(self):
        if self.image is not None:
            mean, ok1 = QInputDialog.getDouble(self, 'Add Gaussian Noise', 'Enter mean:')
            stddev, ok2 = QInputDialog.getDouble(self, 'Add Gaussian Noise', 'Enter standard deviation:')
            if ok1 and ok2:
                noise = cv2.randn(self.image, mean, stddev)
                noisy_image = cv2.add(self.image, noise)
                cv2.imshow('Noisy Image', noisy_image)
                cv2.waitKey(0)
                cv2.destroyAllWindows()

    def convert_to_gray(self):
        if self.image is not None:
            gray_image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
            cv2.imshow('Gray Image', gray_image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

    def histogram_equalization(self):
        if self.image is not None:
            gray_image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
            equalized_image = cv2.equalizeHist(gray_image)
            cv2.imshow('Equalized Image', equalized_image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

    def linear_transform(self):
        if self.image is not None:
            alpha, ok1 = QInputDialog.getDouble(self, 'Linear Transform', 'Enter alpha:')
            beta, ok2 = QInputDialog.getDouble(self, 'Linear Transform', 'Enter beta:')
            if ok1 and ok2:
                transformed_image = cv2.convertScaleAbs(self.image, alpha=alpha, beta=beta)
                cv2.imshow('Transformed Image', transformed_image)
                cv2.waitKey(0)
                cv2.destroyAllWindows()

    def gamma_transform(self):
        if self.image is not None:
            gamma, ok = QInputDialog.getDouble(self, 'Gamma Transform', 'Enter gamma:')
            if ok:
                gamma_corrected_image = cv2.pow(self.image / 255.0, gamma)
                gamma_corrected_image = (gamma_corrected_image * 255).astype('uint8')
                cv2.imshow('Gamma Corrected Image', gamma_corrected_image)
                cv2.waitKey(0)
                cv2.destroyAllWindows()


if __name__ == '__main__':
    app = QApplication(sys.argv)

    main_window = ImageProcessingWindow()
    main_window.show()

    image_app = ImageProcessingApp()
    image_app.show()

    sys.exit(app.exec_())
