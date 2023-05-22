import argparse

import cv2
from PyQt5.QtGui import *
from PyQt5.QtCore import *
from PyQt5.QtWidgets import *
import threading
import sys
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

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

class MainWindow(QTabWidget):
    def __init__(self):
        #初始化界面
        super().__init__()
        self.setWindowTitle('Target detection system')
        self.resize(1200,800)
        #应用图标self.setWindowIcon()
        #图片读取进程
        self.output_size=480
        self.img2predict=""
        self.device='0'
        #初始化视频读取
        self.vid_source='0'
        self.stopEvent=threading.Event()
        self.webcam=True
        self.stopEvent.clear()
        self.model=self.model_load()
        self.initUI()
        self.reset_vid()
        self.flag=False

    '''
    ***模型初始化***
    '''
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
            self.modelc.load_state_dict(torch.load('./weights/Final_LPRNet_model1.pth', map_location=torch.device('cpu')))
            print("load lprnet pretrained model successful!")
            self.modelc.to(self.device).eval()  # 加载模型到gpu
        print("模型加载完成!")
        return model


    '''
    ***界面初始化***
    '''
    def initUI(self):
        # 图片检测子界面
        font_title = QFont('楷体', 16)
        font_main = QFont('楷体', 14)
        # 图片识别界面, 两个按钮，上传图片和显示结果
        img_detection_widget = QWidget()
        img_detection_layout = QVBoxLayout()
        img_detection_title = QLabel("图片识别功能")


        #self.img_detection_lineedit=QLineEdit()
        #self.img_detection_lineedit.setMinimumHeight(20)
        #self.img_detection_lineedit1 = QLineEdit()
        #self.img_detection_lineedit3 = QLineEdit()
        self.img_detection_scroll=QScrollArea()
        self.img_detection_scroll.setMinimumSize(260,70)
        self.scrollAreaWidgetContents = QWidget()
        self.scrollAreaWidgetContents.setMinimumSize(QSize(200,500))
        self.img_detection_slayout = QVBoxLayout(self.scrollAreaWidgetContents)
        #self.img_detection_slayout = QVBoxLayout()

        img_detection_title.setFont(font_title)
        mid_img_widget = QWidget()
        mid_img_layout = QHBoxLayout()
        self.left_img = QLabel()
        self.left_img.setStyleSheet("border:2px groove gray;border-radius:10px;padding:2px 4px;")
        self.left_img.setFixedSize(500,500)
        self.left_img.setText("原图")
        self.right_img = QLabel()
        self.right_img.setStyleSheet("border:2px groove gray;border-radius:10px;padding:2px 4px;")
        self.right_img.setFixedSize(500, 500)
        self.right_img.setText("识别结果")

        self.left_img.setAlignment(Qt.AlignCenter)
        self.right_img.setAlignment(Qt.AlignCenter)
        mid_img_layout.addWidget(self.left_img)
        mid_img_layout.addStretch(0)
        mid_img_layout.addWidget(self.right_img)
        mid_img_widget.setLayout(mid_img_layout)
        up_img_button = QPushButton("上传图片")
        up_img_button.setFixedWidth(800)
        det_img_button = QPushButton("开始检测")
        det_img_button.setFixedWidth(800)
        up_img_button.clicked.connect(self.upload_img)
        det_img_button.clicked.connect(self.detect_img)
        up_img_button.setFont(font_main)
        det_img_button.setFont(font_main)
        up_img_button.setStyleSheet("QPushButton{color:white}"
                                    "QPushButton:hover{background-color: rgb(2,110,180);}"
                                    "QPushButton{background-color:rgb(48,124,208)}"
                                    "QPushButton{border:2px}"
                                    "QPushButton{border-radius:5px}"
                                    "QPushButton{padding:5px 5px}"
                                    "QPushButton{margin:5px 5px}")
        det_img_button.setStyleSheet("QPushButton{color:white}"
                                     "QPushButton:hover{background-color: rgb(2,110,180);}"
                                     "QPushButton{background-color:rgb(48,124,208)}"
                                     "QPushButton{border:2px}"
                                     "QPushButton{border-radius:5px}"
                                     "QPushButton{padding:5px 5px}"
                                     "QPushButton{margin:5px 5px}")
        img_detection_layout.addWidget(img_detection_title, alignment=Qt.AlignCenter)
        img_detection_layout.addWidget(self.img_detection_scroll,alignment=Qt.AlignCenter)
        #self.img_detection_slayout.addWidget(self.img_detection_lineedit,alignment=Qt.AlignCenter)
        #self.img_detection_slayout.addWidget(self.img_detection_lineedit1, alignment=Qt.AlignCenter)
        #self.img_detection_slayout.addWidget(self.img_detection_lineedit3, alignment=Qt.AlignCenter)
        #self.scrollAreaWidgetContents.setLayout(self.img_detection_slayout)
        #self.img_detection_scroll.setWidget(self.scrollAreaWidgetContents)
        img_detection_layout.addWidget(mid_img_widget, alignment=Qt.AlignCenter)
        img_detection_layout.addWidget(up_img_button, alignment=Qt.AlignCenter)
        img_detection_layout.addWidget(det_img_button, alignment=Qt.AlignCenter)
        img_detection_widget.setLayout(img_detection_layout)

        #  视频识别界面
        vid_detection_widget = QWidget()
        vid_detection_layout = QVBoxLayout()
        vid_mid_layout = QHBoxLayout()
        vid_title = QLabel("视频检测功能")

        self.vid_detection_slayout = QVBoxLayout()

        self.vid_lineedit0 = QLineEdit()
        self.vid_lineedit0.setFixedWidth(200)
        self.vid_lineedit1 = QLineEdit()
        self.vid_lineedit1.setFixedWidth(200)
        self.vid_lineedit2 = QLineEdit()
        self.vid_lineedit2.setFixedWidth(200)
        self.vid_lineedit3 = QLineEdit()
        self.vid_lineedit3.setFixedWidth(200)
        self.vid_lineedit4 = QLineEdit()
        self.vid_lineedit4.setFixedWidth(200)
        self.vid_lineedit5 = QLineEdit()
        self.vid_lineedit5.setFixedWidth(200)

        self.vid_tlineedit = QLineEdit()
        self.vid_tlineedit.setFixedWidth(200)
        vid_title.setFont(font_title)
        self.vid_img = QLabel()
        self.vid_img.setFixedSize(750,500)
        #self.vid_img.setPixmap(QPixmap("images/UI/up.jpeg"))
        self.vid_img.setStyleSheet("border:2px groove gray;border-radius:10px;padding:2px 4px;")
        vid_title.setAlignment(Qt.AlignCenter)
        self.vid_img.setAlignment(Qt.AlignCenter)
        self.mp4_detection_btn = QPushButton("视频文件检测")
        self.mp4_detection_btn.setFixedWidth(800)
        self.vid_stop_btn = QPushButton("停止检测")
        self.vid_stop_btn.setFixedWidth(800)
        self.vid_pause_btn = QPushButton("暂停")
        self.vid_pause_btn.setFixedWidth(800)
        #self.vid_pause_btn.setEnabled(False)
        self.mp4_detection_btn.setFont(font_main)
        self.vid_stop_btn.setFont(font_main)
        self.mp4_detection_btn.setStyleSheet("QPushButton{color:white}"
                                             "QPushButton:hover{background-color: rgb(2,110,180);}"
                                             "QPushButton{background-color:rgb(48,124,208)}"
                                             "QPushButton{border:2px}"
                                             "QPushButton{border-radius:5px}"
                                             "QPushButton{padding:5px 5px}"
                                             "QPushButton{margin:5px 5px}")
        self.vid_stop_btn.setStyleSheet("QPushButton{color:white}"
                                        "QPushButton:hover{background-color: rgb(2,110,180);}"
                                        "QPushButton{background-color:rgb(48,124,208)}"
                                        "QPushButton{border:2px}"
                                        "QPushButton{border-radius:5px}"
                                        "QPushButton{padding:5px 5px}"
                                        "QPushButton{margin:5px 5px}")
        self.vid_pause_btn.setStyleSheet("QPushButton{color:white}"
                                        "QPushButton:hover{background-color: rgb(2,110,180);}"
                                        "QPushButton{background-color:rgb(48,124,208)}"
                                        "QPushButton{border:2px}"
                                        "QPushButton{border-radius:5px}"
                                        "QPushButton{padding:5px 5px}"
                                        "QPushButton{margin:5px 5px}")

        self.mp4_detection_btn.clicked.connect(self.open_mp4)
        self.vid_stop_btn.clicked.connect(self.close_vid)
        self.vid_pause_btn.clicked.connect(self.pause_vid)
        # 添加组件到布局上
        self.vid_detection_slayout.addWidget(self.vid_lineedit0, alignment=Qt.AlignCenter)
        self.vid_detection_slayout.addWidget(self.vid_lineedit1, alignment=Qt.AlignCenter)
        self.vid_detection_slayout.addWidget(self.vid_lineedit2, alignment=Qt.AlignCenter)
        self.vid_detection_slayout.addWidget(self.vid_lineedit3, alignment=Qt.AlignCenter)
        self.vid_detection_slayout.addWidget(self.vid_lineedit4, alignment=Qt.AlignCenter)
        self.vid_detection_slayout.addWidget(self.vid_lineedit5, alignment=Qt.AlignCenter)

        vid_mid_layout.addLayout(self.vid_detection_slayout)
        vid_mid_layout.addWidget(self.vid_img, alignment=Qt.AlignCenter)
        vid_mid_layout.addWidget(self.vid_tlineedit, alignment=Qt.AlignCenter)
        vid_detection_layout.addWidget(vid_title)
        vid_detection_layout.addLayout(vid_mid_layout)

        vid_detection_layout.addWidget(self.vid_img, alignment=Qt.AlignCenter)

        vid_detection_layout.addWidget(self.mp4_detection_btn, alignment=Qt.AlignCenter)
        vid_detection_layout.addWidget(self.vid_stop_btn, alignment=Qt.AlignCenter)
        vid_detection_layout.addWidget(self.vid_pause_btn, alignment=Qt.AlignCenter)
        vid_detection_widget.setLayout(vid_detection_layout)

        self.left_img.setAlignment(Qt.AlignCenter)
        self.addTab(img_detection_widget, '图片检测')
        self.addTab(vid_detection_widget, '视频检测')



    '''
    ***上传图片***
    '''
    def upload_img(self):
        # 选择录像文件进行读取
        fileName, fileType = QFileDialog.getOpenFileName(self, 'Choose file', '', '*.jpg *.png *.tif *.jpeg')
        if fileName:
            suffix = fileName.split(".")[-1]
            save_path = osp.join("images/tmp", "tmp_upload." + suffix)
            shutil.copy(fileName, save_path)
            # 应该调整一下图片的大小，然后统一在一起
            im0 = cv2.imread(save_path)
            resize_scale = self.output_size / im0.shape[0]
            im0 = cv2.resize(im0, (0, 0), fx=resize_scale, fy=resize_scale)
            cv2.imwrite("images/tmp/upload_show_result.jpg", im0)
            # self.right_img.setPixmap(QPixmap("images/tmp/single_result.jpg"))
            self.img2predict = fileName
            self.left_img.setPixmap(QPixmap("images/tmp/upload_show_result.jpg"))

    '''
    ***检测图片***
    '''
    def detect_img(self):

        out, source, weights, view_img, save_txt, imgsz = \
            opt.output, opt.source, opt.weights, opt.view_img, opt.save_txt, opt.img_size
        webcam = source == '0' or source.startswith('rtsp') or source.startswith('http') or source.endswith('.txt')
        source=self.img2predict
        print(source)
        if source=="":
            QMessageBox.warning(self,"请上传","请先上传图片再进行检测")
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
                img /= 255.0  #归一化 0 - 255 to 0.0 - 1.0
                # 返回维度 如果图片是三维的
                if img.ndimension() == 3:
                    # 增加维度 batch_size
                    img = img.unsqueeze(0)

                # Inference
                #对每张图片/视频进行前向推理
                self.model.model[6].cv3.conv.register_forward_hook(hook_fn)
                pred = self.model(img, augment=opt.augment)[0]
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

                    #save_path = str(Path(out) / Path(p).name)
                    # txt文件(保存预测框坐标)保存路径
                    txt_path = str(Path(out) / Path(p).stem) + ('_%g' % dataset.frame if dataset.mode == 'video' else '')
                    s += '%gx%g ' % img.shape[2:]  # print string 图片shap（w，h）
                    gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain [whwh] 用于归一化
                    if det is not None and len(det):
                        # Rescale boxes from img_size to im0 size
                        #将预测信息映射回原图
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
                                im0 = plot_one_box(xyxy, im0, label=label, color=(0, 0, 0), line_thickness=3)  # 车牌标签颜色
                                # print(type(im0))
                                self.list = []
                                self.list.append(lb)

                            #print(list)
                            for index in self.list:
                                 print('当前车牌号：%s'% index)
                                 llb=QLineEdit(index)
                                 llb.setMinimumHeight(47)
                                 self.img_detection_slayout.addWidget(llb,alignment=Qt.AlignCenter)
                            self.scrollAreaWidgetContents.setLayout(self.img_detection_slayout)
                            self.img_detection_scroll.setWidget(self.scrollAreaWidgetContents)



                    # Save results (image with detections)
                    if save_img:

                        im0 = np.array(im0)  # 图片转化为 narray
                        cv2.imwrite("images/tmp/single_result.jpg", im0)  # 这个地方的im0必须为narray
                        self.right_img.setPixmap(QPixmap("images/tmp/single_result.jpg"))
                        #print(lb)







    '''
    ### 界面关闭事件 ### 
    '''
    def closeEvent(self, event):
        reply = QMessageBox.question(self,
                                     'quit',
                                     "Are you sure?",
                                     QMessageBox.Yes | QMessageBox.No,
                                     QMessageBox.No)
        if reply == QMessageBox.Yes:
            self.close()
            self.stopEvent.clear()
            event.accept()
        else:
            event.ignore()





    '''
    ### 开启视频文件检测事件 ### 
    '''
    def open_mp4(self):
        fileName, fileType = QFileDialog.getOpenFileName(self, 'Choose file', '', '*.mp4 *.avi')
        if fileName:
            self.mp4_detection_btn.setEnabled(False)
            self.vid_pause_btn.setEnabled(True)
            # self.vid_stop_btn.setEnabled(True)
            self.vid_source = fileName
            self.webcam = False
            self.th = threading.Thread(target=self.detect_vid)
            self.th.start()


    '''
    ### 视频开启事件 ### 
    '''
    def detect_vid(self):
        out, source, weights, view_img, save_txt, imgsz = \
            opt.output, opt.source, opt.weights, opt.view_img, opt.save_txt, opt.img_size
        webcam = source == '0' or source.startswith('rtsp') or source.startswith('http') or source.endswith('.txt')
        source = str(self.vid_source)
        webcam=self.webcam
        # Set Dataloader
        vid_path, vid_writer = None, None
        if webcam:
            save_img = True
            view_img = True
            cudnn.benchmark = True  # set True to speed up constant image size inference
            dataset = LoadStreams(source, img_size=imgsz)
        else:
            save_img = True
            dataset = LoadImages(source, img_size=imgsz)

        # Get names and colors
        names = self.model.module.names if hasattr(self.model, 'module') else self.model.names
        colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(names))]

        # Run inference
        # t0 = time.time()
        img = torch.zeros((1, 3, imgsz, imgsz), device=self.device)  # init img
        _ = self.model(img.half() if self.half else img) if self.device.type != 'cpu' else None  # run once
        # path: 图片/视频的路径
        # img: 进行resize + pad之后的图片
        # img0s: 原尺寸的图片
        # vid_cap: 当读取图片时为None, 读取视频时为视频源
        for path, img, im0s, vid_cap in dataset:
            t1 = time_sync()
            img = torch.from_numpy(img).to(self.device)
            img = img.half() if self.half else img.float()  # 半精度训练uint8 to fp16/32
            img /= 255.0  # 归一化0 - 255 to 0.0 - 1.0
            #如果图片是三维的
            if img.ndimension() == 3:
                img = img.unsqueeze(0)#添加一个维度 batch_size


            # Inference
            # t1 = torch_utils.time_synchronized()
            # 对每张图片/视频进行前向推理
            pred = self.model(img, augment=opt.augment)[0]
            #print(pred.shape)

            # Apply NMS
            # conf_thres 置信度阈值
            # iou_thres iou阈值
            # class 只保留特定类别
            # agnostic_nms 去除不同类别之间的框
            pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes,
                                        agnostic=opt.agnostic_nms)  # 非最大值抑制
            # t2 = torch_utils.time_synchronized()

            # Apply Classifier
            if self.classify:
                pred, plat_num = apply_classifier(pred, self.modelc, img, im0s)


            # Process detections
            # 对每张图片进行处理  将pred映射回原图img0
            for i, det in enumerate(pred):  # detections per image

                pflag=0
                # p: 当前图片/视频的绝对路径 如 F:\yolo_v5\yolov5-U\data\images\bus.jpg
                # s: 输出信息 初始为 ''
                # im0: 原始图片 letterbox + pad 之前的图片
                if webcam:  # batch_size >= 1
                     p, s, im0 = path[i], '%g: ' % i, im0s[i].copy()
                else:
                    p, s, im0 = path, '', im0s

                save_path = str(Path(out) / Path(p).name)
                # txt文件(保存预测框坐标)保存路径
                txt_path = str(Path(out) / Path(p).stem) + (
                    '_%g' % dataset.frame if dataset.mode == 'video' else '')
                # print string 输出信息  图片shape (w, h)
                s += '%gx%g ' % img.shape[2:]
                # normalization gain [whwh]用于归一化
                gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]
                if det is not None and len(det):
                    # Rescale boxes from img_size to im0 size
                    # 将预测信息映射回原图 img0
                    det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()


                    # Write results
                    for de, lic_plat in zip(det, plat_num):
                        # xyxy,conf,cls,lic_plat=de[:4],de[4],de[5],de[6:]
                        *xyxy, conf, cls = de

                        if save_txt:  # Write to file
                            # 将xyxy(左上角 + 右下角)格式转换为xywh(中心的 + 宽高)格式 并除以gn(whwh)做归一化
                            xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(
                                -1).tolist()  # normalized xywh
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
                            im0 = plot_one_box(xyxy, im0, label=label, color=(0, 0, 0), line_thickness=3)  # 车牌标签颜色
                            # print(type(im0))
                            t2 = time_sync()
                            t = t2-t1
                            if pflag==0:
                                if len(lb) < 7:
                                    self.vid_lineedit0.setText("车牌被遮挡或识别错误")
                                else:
                                    self.vid_lineedit0.setText(lb)
                                pflag+=1
                            elif pflag==1:
                                if len(lb) < 7:
                                    self.vid_lineedit1.setText("车牌被遮挡或识别错误")
                                else:
                                    self.vid_lineedit1.setText(lb)
                                pflag+=1
                            elif pflag==2:
                                if len(lb) < 7:
                                    self.vid_lineedit2.setText("车牌被遮挡或识别错误")
                                else:
                                    self.vid_lineedit2.setText(lb)
                                pflag+=1
                            elif pflag==3:
                                if len(lb) < 7:
                                    self.vid_lineedit3.setText("车牌被遮挡或识别错误")
                                else:
                                    self.vid_lineedit3.setText(lb)
                                pflag+=1
                            elif pflag==4:
                                if len(lb) < 7:
                                    self.vid_lineedit4.setText("车牌被遮挡或识别错误")
                                else:
                                    self.vid_lineedit4.setText(lb)
                                pflag+=1
                            elif pflag==5:
                                if len(lb) < 7:
                                    self.vid_lineedit5.setText("车牌被遮挡或识别错误")
                                else:
                                    self.vid_lineedit5.setText(lb)
                                    pflag=0

                            self.vid_tlineedit.setText(str(t))


                            # if len(lb)<7:
                            #     self.vid_lineedit0.setText("车牌不完整")
                            # else:
                            #     self.vid_lineedit0.setText(lb)
                            #     self.vid_tlineedit.setText(str(t))


                # Save results (image with detections)
                if save_img:
                    im0 = np.array(im0)  # 图片转化为 narray
                    cv2.imwrite("images/tmp/single_result_vid.jpg", im0)  # 这个地方的im0必须为narray
                    self.vid_img.setPixmap(QPixmap("images/tmp/single_result_vid.jpg"))
            if cv2.waitKey(25) & self.stopEvent.is_set() == True:
                self.flag=False
                self.stopEvent.clear()
                self.mp4_detection_btn.setEnabled(True)
                self.vid_pause_btn.setEnabled(False)
                self.vid_pause_btn.setText("暂停")
                self.webcam=False
                self.vid_lineedit0.setText("")
                self.vid_lineedit1.setText("")
                self.vid_lineedit2.setText("")
                self.vid_lineedit3.setText("")
                self.vid_lineedit4.setText("")
                self.vid_lineedit5.setText("")
                self.vid_tlineedit.setText("")
                self.reset_vid()
                break
            while self.flag:
                if cv2.waitKey(25) & self.stopEvent.is_set() == True:
                    self.flag=False
                time.sleep(1)


        # if save_txt or save_img:
        #     print('Results saved to %s' % os.getcwd() + os.sep + out)
        #     if platform == 'darwin':  # MacOS
        #         os.system('open ' + save_path)


    '''
    ### 界面重置事件 ### 
    '''
    def reset_vid(self):
        self.mp4_detection_btn.setEnabled(True)
        #self.vid_img.setPixmap(QPixmap("images/UI/up.jpeg"))
        self.vid_source = '0'
        self.webcam = False


    '''
    ### 视频重置事件 ### 
    '''
    def close_vid(self):
        self.stopEvent.set()
        self.reset_vid()


    '''
    ### 视频暂停事件 ### 
    '''
    def pause_vid(self):
        if self.vid_pause_btn.text()=="暂停":
            self.vid_pause_btn.setText("继续")
            self.flag=True
        else:
            self.vid_pause_btn.setText("暂停")
            self.flag=False

if __name__ == "__main__":
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
    app = QApplication(sys.argv)
    mainWindow = MainWindow()
    apply_stylesheet(app,theme='dark_teal.xml')
    mainWindow.show()
    sys.exit(app.exec_())
