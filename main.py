import sys
import cv2
import numpy as np
from PyQt5 import uic
from PyQt5.QtWidgets import QApplication, QMainWindow, QFileDialog, QInputDialog, QMessageBox
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt


import argparse
import threading
import os.path as osp
import torch.backends.cudnn as cudnn
from models.experimental import *
from utils.datasets import *
from utils.utils import *
from models.LPRNet import *
from utils.torch_utils import time_sync
from qt_material import apply_stylesheet
import os
from weights.obtain_feature_map import hook_fn


class ImageProcessingWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('Image Processing')
        # 从文件中加载UI定义
        self.ui = uic.loadUi("main.ui")
        self.output_size = 480

        # 加载初始提醒图片
        pix = QPixmap(r'init.jpg')
        self.image = cv2.imread(r'init.jpg')
        self.ui.image_label.setPixmap(pix.scaled(self.ui.image_label.size(), Qt.KeepAspectRatio))
        self.ui.b_show_image.clicked.connect(self.show_image)
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
        self.ui.b_gamma_transform.clicked.connect(self.apply_gamma_transform)
        self.ui.b_histogram_equalization.clicked.connect(self.apply_histogram_equalization)

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
        self.ui.b_set_gaussian.clicked.connect(self.set_gaussian)
        self.ui.b_set_laplacian.clicked.connect(self.set_laplacian)

        # 加载模型
        self.model = self.model_load()
        # 检测图片文件路径
        self.img2predict=""
        self.ui.b_sharpen.clicked.connect(self.upload_img)
        # 打开需要检测的图片
        self.ui.b_choose_pic.clicked.connect(self.upload_img)
        self.ui.b_detect.clicked.connect(self.detect_img)


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

    def show_image(self):
        cv2.imshow("Result", self.image)

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

    # 图像裁剪
    def crop_image(self):
        if self.image is not None:
            roi = cv2.selectROI('Please choose the scope, then press enter!', self.image, fromCenter=False,
                                showCrosshair=True)
            if roi != (0, 0, 0, 0):
                cropped_image = self.image[int(roi[1]):int(roi[1] + roi[3]), int(roi[0]):int(roi[0] + roi[2])]
                self.image = cropped_image
                self.show_cv_image(self.image)
                cv2.waitKey(0)
                cv2.destroyAllWindows()

    # 色彩图转灰度图
    def convert_to_gray(self):
        if self.image is not None:
            gray_image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
            self.image = gray_image
            self.show_cv_image(self.image)
            # cv2.imshow('Gray Image', gray_image)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()

    # 高斯滤波
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

    # 直方图均值化
    def apply_histogram_equalization(self):
        if self.image is not None:
            gray_image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
            equalized_image = cv2.equalizeHist(gray_image)
            colored_equalized_image = cv2.cvtColor(equalized_image, cv2.COLOR_GRAY2BGR)
            self.image = colored_equalized_image
            self.show_cv_image(colored_equalized_image)

    # 线性变化
    def apply_linear_transform(self):
        if self.image is not None:
            alpha, ok = QInputDialog.getDouble(self, 'Linear Transform', 'Enter alpha value:')
            if ok:
                beta, ok = QInputDialog.getDouble(self, 'Linear Transform', 'Enter beta value:')
                if ok:
                    transformed_image = cv2.convertScaleAbs(self.image, alpha=alpha, beta=beta)
                    self.show_cv_image(transformed_image)

    # 伽马变换
    def apply_gamma_transform(self):
        if self.image is not None:
            gamma, ok = QInputDialog.getDouble(self, 'Gamma Transform', '输入伽马值:')
            if ok:
                gamma_corrected = np.power(self.image / 255.0, gamma)
                self.image = (gamma_corrected * 255).astype(np.uint8)
                self.show_cv_image(self.image)

    def conv(self, conv_k, stride):
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

    def set_laplacian(self):
        self.ui.k1.setText('1')
        self.ui.k2.setText('1')
        self.ui.k3.setText('1')
        self.ui.k4.setText('1')
        self.ui.k5.setText('-8')
        self.ui.k6.setText('1')
        self.ui.k7.setText('1')
        self.ui.k8.setText('1')
        self.ui.k9.setText('1')

    def set_gaussian(self):
        self.ui.k1.setText('0.1')
        self.ui.k2.setText('0.1')
        self.ui.k3.setText('0.1')
        self.ui.k4.setText('0.1')
        self.ui.k5.setText('0.2')
        self.ui.k6.setText('0.1')
        self.ui.k7.setText('0.1')
        self.ui.k8.setText('0.1')
        self.ui.k9.setText('0.1')

    @torch.no_grad()
    def model_load(self):
        # yolo = YOLO()
        out, source, weights, view_img, save_txt, imgsz = \
            opt.output, opt.source, opt.weights, opt.view_img, opt.save_txt, opt.img_size
        webcam = source == '0' or source.startswith('rtsp') or source.startswith('http') or source.endswith('.txt')

        # Initialize
        self.device = torch_utils.select_device(opt.device)
        if os.path.exists(out):
            shutil.rmtree(out)  # delete output folder 递归删除
        os.makedirs(out)  # make new output folder
        self.half = self.device.type != 'cpu'  # half precision only supported on CUDA

        # Load model
        model = attempt_load(weights, map_location=self.device)  # load FP32 model
        imgsz = check_img_size(imgsz, s=model.stride.max())  # check img_size
        if self.half:
            model.half()  # to FP16

        # Second-stage classifier
        self.classify = True
        if self.classify:
            self.modelc = LPRNet(lpr_max_len=8, phase=False, class_num=len(CHARS), dropout_rate=0).to(self.device)
            self.modelc.load_state_dict(
                torch.load('./weights/Final_LPRNet_model1.pth', map_location=torch.device('cpu')))
            print("load lprnet pretrained model successful!")
            self.modelc.to(self.device).eval()  # 加载模型到gpu
        print("模型加载完成!")
        return model

    def upload_img(self):
        # 选择录像文件进行读取
        file_dialog = QFileDialog()
        fileName, fileType = QFileDialog.getOpenFileName(self, 'Choose file', '', '*.jpg *.png *.tif *.jpeg')
        pix = QPixmap(fileName)
        self.ui.ori_pic.setPixmap(pix.scaled(self.ui.image_label.size(), Qt.KeepAspectRatio))
        if fileName:
            suffix = fileName.split(".")[-1]
            # save_path = "images/tmp/"+"tmp_upload." + suffix
            save_path = osp.join("images/tmp", "tmp_upload." + suffix)
            shutil.copy(fileName, save_path)
            print(save_path)
            # 应该调整一下图片的大小，然后统一在一起
            im0 = cv2.imread(save_path)
            # cv2.imshow("Result", im0)
            resize_scale = self.output_size / im0.shape[0]
            im0 = cv2.resize(im0, (0, 0), fx=resize_scale, fy=resize_scale)

            cv2.imwrite("images/tmp/upload_show_result.jpg", im0)
            # self.right_img.setPixmap(QPixmap("images/tmp/single_result.jpg"))
            self.img2predict = fileName
            # self.left_img.setPixmap(QPixmap("images/tmp/upload_show_result.jpg"))

    def detect_img(self):

        out, source, weights, view_img, save_txt, imgsz = \
            opt.output, opt.source, opt.weights, opt.view_img, opt.save_txt, opt.img_size
        webcam = source == '0' or source.startswith('rtsp') or source.startswith('http') or source.endswith('.txt')
        source = self.img2predict

        if source == "":
            QMessageBox.warning(self, "请上传", "请先上传图片再进行检测")
        else:
            save_img = True
            dataset = LoadImages(source, img_size=imgsz)

            # 得到数据集的所有类的类名
            names = self.model.module.names if hasattr(self.model, 'module') else self.model.names
            colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(names))]

            # Run inference
            # t0 = time.time()
            img = torch.zeros((1, 3, imgsz, imgsz), device=self.device)  # init img
            _ = self.model(img.half() if self.half else img) if self.device.type != 'cpu' else None  # run once
            for path, img, im0s, vid_cap in dataset:
                t1 = time_sync
                # 数组转化张量
                img = torch.from_numpy(img).to(self.device)
                img = img.half() if self.half else img.float()  # 半精度训练 uint8 to fp16/32
                img /= 255.0  # 归一化 0 - 255 to 0.0 - 1.0
                # 返回维度 如果图片是三维的
                if img.ndimension() == 3:
                    # 增加维度 batch_size
                    img = img.unsqueeze(0)

                # Inference
                # 对每张图片/视频进行前向推理
                self.model.model[0].conv.register_forward_hook(hook_fn('m[0]'))
                self.model.model[1].conv.register_forward_hook(hook_fn('m[1]'))
                self.model.model[2].cv1.conv.register_forward_hook(hook_fn('m[2]'))
                self.model.model[5].conv.register_forward_hook(hook_fn('m[5]'))
                pred = self.model(img, augment=opt.augment)[0]
                self.model.model[0].conv.register_forward_hook(hook_fn('m[0]')).remove()
                self.model.model[1].conv.register_forward_hook(hook_fn('m[1]')).remove()
                self.model.model[2].cv1.conv.register_forward_hook(hook_fn('m[2]')).remove()
                self.model.model[5].conv.register_forward_hook(hook_fn('m[5]')).remove()
                print(pred.shape)
                # Apply NMS
                # conf_thres 置信度阈值
                # iou_thres iou阈值
                # class 只保留特定类别
                # agnostic_nms 去除不同类别之间的框
                pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes,
                                           agnostic=opt.agnostic_nms)  # 非最大值抑制

                # Apply Classifier
                if self.classify:
                    # img: 进行resize + pad之后的图片
                    # img0s: 原尺寸的图片
                    pred, plat_num = apply_classifier(pred, self.modelc, img, im0s)
                # Process detections
                # 对每张图片进行处理  将pred映射回原图img0
                # p: 当前图片  的绝对路径
                # s: 输出信息 初始为 ''
                # im0: 原始图片
                for i, det in enumerate(pred):  # detections per image
                    if webcam:  # batch_size >= 1
                        p, s, im0 = path[i], '%g: ' % i, im0s[i].copy()
                    else:
                        p, s, im0 = path, '', im0s

                    # save_path = str(Path(out) / Path(p).name)
                    # txt文件(保存预测框坐标)保存路径
                    txt_path = str(Path(out) / Path(p).stem) + (
                        '_%g' % dataset.frame if dataset.mode == 'video' else '')
                    s += '%gx%g ' % img.shape[2:]  # print string 图片shap（w，h）
                    gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain [whwh] 用于归一化
                    if det is not None and len(det):
                        # Rescale boxes from img_size to im0 size
                        # 将预测信息映射回原图
                        det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                        # Write results

                        for de, lic_plat in zip(det, plat_num):
                            # xyxy,conf,cls,lic_plat=de[:4],de[4],de[5],de[6:]
                            *xyxy, conf, cls = de

                            if save_txt:  # Write to file
                                # 将xyxy(左上角 + 右下角)格式转换为xywh(中心的 + 宽高)格式 并除以gn(whwh)做归一化
                                xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()
                                with open(txt_path + '.txt', 'a') as f:
                                    f.write(('%g ' * 5 + '\n') % (cls, xywh))  # label format

                            if save_img or view_img:  # Add bbox to image
                                # label = '%s %.2f' % (names[int(cls)], conf)
                                lb = ""
                                for a, i in enumerate(lic_plat):
                                    # if a ==0:
                                    #     continue
                                    lb += CHARS[int(i)]
                                label = '%s %.2f' % (lb, conf)
                                # im0=plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=3)
                                im0 = plot_one_box(xyxy, im0, label=label, color=(0, 0, 0),
                                                   line_thickness=3)  # 车牌标签颜色
                                # print(type(im0))
                                self.list = []
                                self.list.append(lb)

                            for index in self.list:
                                print('当前车牌号：%s' % index)
                                self.ui.license_plate.setText(index)
                                # llb = QLineEdit(index)
                                # llb.setMinimumHeight(47)
                                # self.img_detection_slayout.addWidget(llb, alignment=Qt.AlignCenter)
                            # self.scrollAreaWidgetContents.setLayout(self.img_detection_slayout)
                            # self.img_detection_scroll.setWidget(self.scrollAreaWidgetContents)

                    # Save results (image with detections)
                    if save_img:
                        im0 = np.array(im0)  # 图片转化为 narray
                        cv2.imwrite("images/tmp/single_result.jpg", im0)  # 这个地方的im0必须为narray
                        self.ui.mid_img_1.setPixmap(QPixmap("m[0].png").scaled(self.ui.image_label.size(), Qt.KeepAspectRatio))
                        self.ui.mid_img_2.setPixmap(
                            QPixmap("m[1].png").scaled(self.ui.image_label.size(), Qt.KeepAspectRatio))
                        self.ui.mid_img_3.setPixmap(
                            QPixmap("m[2].png").scaled(self.ui.image_label.size(), Qt.KeepAspectRatio))
                        self.ui.mid_img_4.setPixmap(
                            QPixmap("m[5].png").scaled(self.ui.image_label.size(), Qt.KeepAspectRatio))
                        pix = QPixmap("images/tmp/single_result.jpg")
                        self.ui.res_pic.setPixmap(pix.scaled(self.ui.image_label.size(), Qt.KeepAspectRatio))
                        # self.right_img.setPixmap(QPixmap("images/tmp/single_result.jpg"))

                        # print(lb)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='./weights/last.pt', help='model.pt path(s)')
    parser.add_argument('--source', type=str, default='./inference/images/', help='source')  # file/folder, 0 for webcam
    parser.add_argument('--output', type=str, default='inference/output', help='output folder')  # output folder
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.4, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.5, help='IOU threshold for NMS')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='display results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--update', action='store_true', help='update all models')
    opt = parser.parse_args()

    main_window = ImageProcessingWindow()
    main_window.ui.show()

    sys.exit(app.exec_())
