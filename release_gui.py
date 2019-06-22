import sys
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
import cv2
import numpy as np
import os
from PIL import Image
import imagehash
from imutils.perspective import four_point_transform
import calibration_opencv
import time


class App(QMainWindow):
    def __init__(self):
        super().__init__()
        self.title = 'VKR_program_rangefinder'
        self.setWindowTitle(self.title)
        self.setWindowIcon(QIcon(os.path.join('D:\learn\8 sem\diplom\diplom','resources\pictures\eye.png')))
        self.left = 15
        self.top = 35
        self.width = 1276
        self.height = 550

        self.setGeometry(self.left, self.top, self.width, self.height)
        self.table_widget = MainWin(self)
        self.setCentralWidget(self.table_widget)

        self.show()


class MainWin(QWidget):
    def __init__(self, parent):
        super(QWidget, self).__init__(parent)

        self.main_layout = QVBoxLayout()
        self.buttons = QHBoxLayout()

        self.tabs = QTabWidget()
        self.tab1 = QWidget()
        self.tab2 = QWidget()
        self.tab3 = QWidget()

        self.tabs.addTab(self.tab1, 'Видеопоток с камер')
        self.tabs.addTab(self.tab2, 'Преобразованный видеопоток')
        self.tabs.addTab(self.tab3, 'Карта диспарантностей')

        path_stdb = os.path.join('D:\diplom\diplom','resources\pictures\please-stand-by.jpg')
        self.standby = self.get_qimage(cv2.imread(path_stdb))

        self.t1_left = QLabel()
        self.t1_right = QLabel()

        self.t2_left = QLabel()
        self.t2_right = QLabel()

        self.t3_center = QLabel()


        self.start_button = QPushButton('Start')
        self.start_button.clicked.connect(self.start)
        self.stop_button = QPushButton('Stop')
        self.stop_button.clicked.connect(self.stop)

        self.buttons.addWidget(self.start_button)
        self.buttons.addWidget(self.stop_button)

        self.image1 = QPixmap()
        self.image2 = QPixmap()
        self.image3 = QPixmap()
        self.image4 = QPixmap()
        self.image5 = QPixmap()

        self.tab1_ui()
        self.tab2_ui()
        self.tab3_ui()

        self.main_layout.addWidget(self.tabs)
        self.main_layout.addLayout(self.buttons)
        self.setLayout(self.main_layout)

    def tab1_ui(self):
        layout = QVBoxLayout()
        lbls = QHBoxLayout()

        lbls.addWidget(self.t1_left)
        lbls.addWidget(self.t1_right)

        self.t1_left.setPixmap(QPixmap.fromImage(self.standby))
        self.t1_right.setPixmap(QPixmap.fromImage(self.standby))

        text_line = QLabel('Исходные изображения')
        text_line.setFont(QFont('Times', 15))  # Изменение шрифта и размера

        layout.addWidget(text_line, alignment=Qt.AlignCenter)
        layout.addLayout(lbls)
        self.tab1.setLayout(layout)

    def tab2_ui(self):
        layout = QVBoxLayout()
        lbls = QHBoxLayout()

        lbls.addWidget(self.t2_left)
        lbls.addWidget(self.t2_right)

        self.t2_left.setPixmap(QPixmap.fromImage(self.standby))
        self.t2_right.setPixmap(QPixmap.fromImage(self.standby))

        text_line = QLabel('Отректифицированные изображения')
        text_line.setFont(QFont('Times', 15))

        layout.addWidget(text_line, alignment=Qt.AlignHCenter)
        layout.addLayout(lbls)
        self.tab2.setLayout(layout)

    def tab3_ui(self):
        layout = QVBoxLayout()

        # text_line = QLabel('Карта диспарантностей')

        self.t3_center.setPixmap(QPixmap.fromImage(self.standby))

        #layout.addWidget(text_line, alignment=Qt.AlignHCenter)
        layout.addWidget(self.t3_center, alignment=Qt.AlignHCenter)
        self.tab3.setLayout(layout)

    @staticmethod
    def get_qimage(img):
        height, width, channel = img.shape
        bytes = channel * width
        convert_image = QImage(img.data, width, height, bytes, QImage.Format_RGB888) \
            .scaled(640, 480, Qt.KeepAspectRatio)
        return convert_image.rgbSwapped()

    def start(self):
        self.thread = QThread(self)
        self.video = VideoStream(self)
        self.video.moveToThread(self.thread)
        self.thread.started.connect(self.video.run)
        self.video.onchange_tab1_l.connect(self.change_tab1_lbl_l)
        self.video.onchange_tab1_r.connect(self.change_tab1_lbl_r)
        self.video.onchange_tab2_l.connect(self.change_tab2_lbl_l)
        self.video.onchange_tab2_r.connect(self.change_tab2_lbl_r)
        self.video.onchange_tab3.connect(self.change_tab3_lbl)
        self.thread.start()

    def stop(self):
        self.video.active = False
        self.clear_lbls()
        self.thread.quit()
        self.thread.wait()

    def clear_lbls(self):
        self.video.left_cam.release()
        self.video.right_cam.release()
        self.t1_left.setPixmap(QPixmap.fromImage(self.standby))
        self.t1_right.setPixmap(QPixmap.fromImage(self.standby))
        self.t2_left.setPixmap(QPixmap.fromImage(self.standby))
        self.t2_right.setPixmap(QPixmap.fromImage(self.standby))
        self.t3_center.setPixmap(QPixmap.fromImage(self.standby))

    @pyqtSlot(QImage)
    def change_tab1_lbl_l(self, im):
        self.image1 = QPixmap.fromImage(im)
        self.t1_left.setPixmap(self.image1)

    @pyqtSlot(QImage)
    def change_tab1_lbl_r(self, im):
        self.image2 = QPixmap.fromImage(im)
        self.t1_right.setPixmap(self.image2)

    @pyqtSlot(QImage)
    def change_tab2_lbl_l(self, im):
        self.image3 = QPixmap.fromImage(im)
        self.t2_left.setPixmap(self.image3)


    @pyqtSlot(QImage)
    def change_tab2_lbl_r(self, im):
        self.image4 = QPixmap.fromImage(im)
        self.t2_right.setPixmap(self.image4)

    @pyqtSlot(QImage)
    def change_tab3_lbl(self, im):
        self.image5 = QPixmap.fromImage(im)
        self.t3_center.setPixmap(self.image5)


class VideoStream(QObject):
    onchange_tab1_l = pyqtSignal(QImage)
    onchange_tab1_r = pyqtSignal(QImage)
    onchange_tab2_l = pyqtSignal(QImage)
    onchange_tab2_r = pyqtSignal(QImage)
    onchange_tab3 = pyqtSignal(QImage)

    def __init__(self,queue):
        super().__init__()
        self.cam = cv2.VideoCapture(0)
        self.left_cam = cv2.VideoCapture(1)
        self.right_cam = cv2.VideoCapture(2)

        path1 = os.path.join('D:\diplom\diplom','resources\pictures','road_signs\pedestrian_cross_ref_0.jpg')
        path2 = os.path.join('D:\diplom\diplom','resources\pictures','road_signs\pedestrian_cross_ref_90.jpg')
        path3 = os.path.join('D:\diplom\diplom','resources\pictures','road_signs\pedestrian_cross_ref_180.jpg')
        path4 = os.path.join('D:\diplom\diplom','resources\pictures','road_signs\pedestrian_cross_ref_270.jpg')
        path5 = os.path.join('D:\diplom\diplom', 'resources\pictures', 'road_signs', 'brick.jpg')
        path6 = os.path.join('D:\diplom\diplom', 'resources\pictures', 'road_signs', 'brick_45.jpg')
        path7 = os.path.join('D:\diplom\diplom', 'resources\pictures', 'road_signs', 'brick_90.jpg')
        path8 = os.path.join('D:\diplom\diplom', 'resources\pictures', 'road_signs', 'brick_135.jpg')
        self.pedestrian_ref_0 = imagehash.average_hash(Image.open(path1))
        self.pedestrian_ref_90 = imagehash.average_hash(Image.open(path2))
        self.pedestrian_ref_180 = imagehash.average_hash(Image.open(path3))
        self.pedestrian_ref_270 = imagehash.average_hash(Image.open(path4))
        self.brick_0 = imagehash.average_hash(Image.open(path5))
        self.brick_45 = imagehash.average_hash(Image.open(path6))
        self.brick_90 = imagehash.average_hash(Image.open(path7))
        self.brick_135 = imagehash.average_hash(Image.open(path8))

    @staticmethod
    def get_frame(frame, onechannel):
        if onechannel:
            height, width = frame.shape
            convert_image = QImage(frame.data, width, height, QImage.Format_Grayscale8)
        else:
            height, width, channel = frame.shape
            bytes = channel * width
            convert_image = QImage(frame.data, width, height, bytes, QImage.Format_RGB888).rgbSwapped()
        return convert_image

    @staticmethod
    def hamming_distance(chaine1, chaine2):
        return sum(c1 != c2 for c1, c2 in zip(chaine1, chaine2))

    def hash_compare_circle(self,fragment):
        segment = Image.fromarray(fragment)
        frag_hash = imagehash.average_hash(segment)
        angle0 = self.hamming_distance(str(frag_hash), str(self.brick_0))
        angle1 = self.hamming_distance(str(frag_hash), str(self.brick_45))
        angle2 = self.hamming_distance(str(frag_hash), str(self.brick_90))
        angle3 = self.hamming_distance(str(frag_hash), str(self.brick_135))
        #print(angle0, angle1, angle2, angle3)
        if angle0 <= 10 or angle1 <= 10 or angle2 <= 10 or angle3 <= 10:
            return True
        else:
            return False

    def hash_compare_square(self, fragment):
        segment = Image.fromarray(fragment)
        frag_hash = imagehash.average_hash(segment)
        angle0 = self.hamming_distance(str(frag_hash), str(self.pedestrian_ref_0))
        angle1 = self.hamming_distance(str(frag_hash), str(self.pedestrian_ref_90))
        angle2 = self.hamming_distance(str(frag_hash), str(self.pedestrian_ref_180))
        angle3 = self.hamming_distance(str(frag_hash), str(self.pedestrian_ref_270))
        # print(angle0, angle1, angle2, angle3)
        if angle0 <= 10 or angle1 <= 10 or angle2 <= 10 or angle3 <= 10:
            return True
        else:
            return False

    def search_signs(self, image):
        boxes = []
        circles = []

        # нижний и верхний цветовой порог
        hsv_min = np.array((0, 92, 86), np.uint8)
        hsv_max = np.array((255, 255, 255), np.uint8)

        # переводим в цветовое пространство hsv и размываем
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        blur = cv2.medianBlur(hsv, 3)

        # выделяем цветовой диапазон нужных нам знаков
        thresh = cv2.inRange(blur, hsv_min, hsv_max)

        # mask = cv2.erode(thresh, None, iterations=3)
        # mask = cv2.dilate(mask, None, iterations=5)

        # выделяем контуры детектором Канни
        canny_detector = cv2.Canny(thresh, 25, 200, apertureSize=3)

        # находим контуры на изображении после детектора
        contours = cv2.findContours(canny_detector, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[0]

        # проходимся по списку найденных контуров
        # и пытаемся вписать в них прямоугольник
        # если это удаётся, считаем площадь прямоугольника
        # если она является максимальной на текущий момент
        # то сохраняем координаты вершин фигуры

        if len(contours) > 0:
            # проход по списку контуров, в поиске квадратных знаков
            for cnt in contours:
                rect = cv2.minAreaRect(cnt)  # объект найденного квадрата
                width, height = rect[1]
                area = width * height  # площадь
                if area > 2000:  # ограничение по площади
                    box = cv2.boxPoints(rect)
                    box = np.int0(box)
                    fragment = four_point_transform(image, [box][0])  # вырезаем по координатам
                    flag = self.hash_compare_square(fragment)  # проверяем, знак ли, возвращает True или False
                    if flag:
                        boxes.append(rect)  # если знак, то в список найденных

            for cnt in contours:
                # проход по списку контуров, в поиске круглых знаков
                center, radius = cv2.minEnclosingCircle(cnt)  # вписываем круг
                centerx = int(center[0])
                centery = int(center[1])
                radius = int(radius)
                area = 3.14 * radius * radius
                if area > 2000:  # ограничение по площади
                    rect = cv2.minAreaRect(cnt)  # вписываем туда квадрат
                    box = cv2.boxPoints(rect)  # объект квадрата
                    box = np.int0(box)
                    fragment = four_point_transform(image, [box][0])  # вырезаем квадратную область
                    flag = self.hash_compare_circle(fragment)  # проверяем, знак ли, возвращает True или False

                    if flag:
                        circles.append([centerx, centery, radius])  # если знак, то в список найденных

        return boxes, circles

    def run(self):
        path1 = os.path.join('D:\diplom\diplom', 'resources\configs\left_cam', 'left_cam_m.npy')
        path2 = os.path.join('D:\diplom\diplom', 'resources\configs\left_cam', 'left_cam_dist.npy')
        path3 = os.path.join('D:\diplom\diplom', 'resources\configs', 'right_cam', 'right_cam_m.npy')
        path4 = os.path.join('D:\diplom\diplom', 'resources\configs', 'right_cam', 'right_cam_dist.npy')
        path5 = os.path.join('D:\diplom\diplom', 'resources\configs', 'r_mat.npy')
        path6 = os.path.join('D:\diplom\diplom', 'resources\configs', 't_mat.npy')
        path7 = os.path.join('D:\diplom\diplom', 'resources\configs', 'e_mat.npy')
        path8 = os.path.join('D:\diplom\diplom', 'resources\configs', 'f_mat.npy')

        if not os.path.exists(path1):
            calibration_opencv.calibrate_cameras()

        print('Load settings')
        cm_left = np.load(path1)
        dc_left = np.load(path2)
        cm_right = np.load(path3)
        dc_right = np.load(path4)
        r = np.load(path5)
        t = np.load(path6)
        e = np.load(path7)
        f = np.load(path8)

        self.active = True

        while self.active:
            box_center, circle_center = None, None
            max_rect_left, max_rect_right = None, None
            max_circle_left, max_circle_right = None, None
            circle, box = False, False

            # f1, frame_L = self.left_cam.read()
            # f2, frame_R = self.right_cam.read()

            frame_L = cv2.imread(os.path.join('D:\diplom\diplom','resources\pictures','test1\left_62cm.jpg'))
            frame_R = cv2.imread(os.path.join('D:\diplom\diplom','resources\pictures','test1\lright_62cm.jpg'))

            # if not f1 or not f2:
            #     print('Failed to load stream')
            #     self.active = False

            box_left, circle_left = self.search_signs(frame_L)
            box_right, circle_right = self.search_signs(frame_R)
            # print(len(box_left), ' ', len(box_right))

            # если найдены квадраты в левом и правом
            if len(box_left) > 0 and len(box_right) and len(box_left) == len(box_right):
                max_area_left = 0
                max_area_right = 0
                for left, right in zip(box_left, box_right):  # идём по списку левых квадратов и правых
                    width, height = left[1]
                    area_l = width * height  # площадь
                    if area_l > max_area_left:
                        max_rect_left = left

                    width, height = right[1]
                    area_r = width * height  # площадь
                    if area_r > max_area_right:
                        max_rect_right = right

                # координаты левого
                box_l = cv2.boxPoints(max_rect_left)
                box_l = np.int0(box_l)
                cv2.drawContours(frame_L, [box_l], 0, (0, 255, 0), 2)  # рисуем квадрат на левом
                center_left = max_rect_left[0]

                # координаты правого
                box_r = cv2.boxPoints(max_rect_right)
                box_r = np.int0(box_r)
                cv2.drawContours(frame_R, [box_r], 0, (0, 255, 0), 2)  # рисуем квадрат на правом
                center_right = max_rect_right[0]

                center_x = int((center_left[0] + center_right[0])) // 2  # совмещаем координаты
                center_y = int((center_left[1] + center_right[1])) // 2  # найденных квадратов

                box_center = [center_x, center_y]  # точка для определения расстояния
                box = True

            # если найдены круги в левом и правом
            if circle_left and circle_right and len(circle_left) == len(circle_right):
                max_area_left = 0
                max_area_right = 0
                for left, right in zip(circle_left, circle_right):  # идём по списку левых и правых кружков
                    left_r = left[2]
                    area = 3.14 * left_r * left_r
                    if area > max_area_left:
                        max_circle_left = left

                    right_r = right[2]
                    area = 3.14 * right_r * right_r
                    if area > max_area_right:
                        max_circle_right = right

                # левый
                center_left = (max_circle_left[0], max_circle_left[1])  # центр
                rad_left = max_circle_left[2]  # радиус
                cv2.circle(frame_L, center_left, rad_left, (255, 208, 0), 2)  # рисуем кружок на левом

                # правый
                center_right = (max_circle_right[0], max_circle_right[1])  # центр
                rad_right = max_circle_right[2]  # радиус
                cv2.circle(frame_R, center_right, rad_right, (255, 208, 0), 2)  # рисуем кружок на левом

                center_x = (center_left[0] + center_right[0]) // 2  # совмещаем координаты
                center_y = (center_left[1] + center_right[1]) // 2  # найденных кружков

                circle_center = [center_x, center_y]  # точка для определения расстояния
                circle = True

            # корректировка изображений
            h, w = frame_L.shape[:2]

            # print('Start undistortion...')
            new_left_mtx, roi_l = cv2.getOptimalNewCameraMatrix(cm_left, dc_left, (w, h), 1, (w, h))
            new_right_mtx, roi_r = cv2.getOptimalNewCameraMatrix(cm_right, dc_right, (w, h), 1, (w, h))

            # print('Start stereoRectify...')
            # r_left, r_right, pm_left, pm_right, q, roi_left, roi_right = cv2.stereoRectify(cm_left, dc_left, cm_right, dc_right,
            #                                                                                (w, h), r, t)

            # undistort left
            mapx1, mapy1 = cv2.initUndistortRectifyMap(cm_left, dc_left, r, new_left_mtx, (w, h), 5)
            dst_left = cv2.remap(frame_L, mapx1, mapy1, cv2.INTER_LINEAR)
            # x, y, w, h = roi_l
            # dst_left = dst_left[y:y + h, x:x + w]

            # undistort right
            mapx2, mapy2 = cv2.initUndistortRectifyMap(cm_right, dc_right, r, new_right_mtx, (w, h), 5)
            dst_right = cv2.remap(frame_R, mapx2, mapy2, cv2.INTER_LINEAR)
            # x, y, w, h = roi_r
            # dst_right = dst_right[y:y + h, x:x + w]
            # print('Done!')

            dst_left = cv2.cvtColor(dst_left, cv2.COLOR_BGR2GRAY)
            dst_right = cv2.cvtColor(dst_right, cv2.COLOR_BGR2GRAY)

            # строим карту смещений
            matcher = cv2.StereoSGBM_create(minDisparity=36,
                                            numDisparities=112,
                                            blockSize=11,
                                            P1=8 * 3 * 11 ** 2,
                                            # wsize default 3; 5; 7 for SGBM reduced size image; 15 for SGBM full size image (1300px and above); 5 Works nicely
                                            P2=32 * 3 * 11 ** 2,
                                            disp12MaxDiff=60,
                                            uniquenessRatio=0,
                                            speckleWindowSize=223,
                                            speckleRange=100,
                                            mode=False)
            displ = matcher.compute(dst_left, dst_right).astype(np.float32) / 16
            filteredImg = cv2.normalize(src=displ, dst=displ, beta=0, alpha=255, norm_type=cv2.NORM_MINMAX,
                                        dtype=cv2.CV_8U)

            if box:
                x1, y1 = max_rect_left[0]
                x2, y2 = max_rect_right[0]

                dot = filteredImg[box_center[0]][box_center[1]]
                distance = round((0.004*0.072)/(0.263 * dot) * 10000000,2)  # (focal_l * base) / (size_of_pix * dispar)
                cv2.putText(frame_L, str(distance) + 'cm', (int(x1), int(y1)), cv2.FAST_FEATURE_DETECTOR_FAST_N, 1, (20, 28, 255),
                            lineType=cv2.LINE_AA)
                cv2.putText(frame_R, str(distance) + 'cm', (int(x2), int(y2)), cv2.FAST_FEATURE_DETECTOR_FAST_N, 1, (20, 28, 255),
                            lineType=cv2.LINE_AA)
            elif circle:
                x1, y1 = max_circle_left[0], max_circle_left[1]
                x2, y2 = max_circle_right[0], max_circle_right[1]

                dot = filteredImg[circle_center[0]][circle_center[1]]
                distance = round((0.004*0.072)/(0.263 * dot) * 10000000,2)   # (focal_l * base) / (size_of_pix * dispar)
                cv2.putText(frame_L, str(distance) + 'cm', (int(x1), int(y1)), cv2.FAST_FEATURE_DETECTOR_FAST_N, 1.0, (39, 255, 255),
                            lineType=cv2.LINE_AA)
                cv2.putText(frame_R, str(distance) + 'cm', (int(x2), int(y2)), cv2.FAST_FEATURE_DETECTOR_FAST_N, 1.0, (39, 255, 255),
                            lineType=cv2.LINE_AA)
            if not self.active:
                break

            self.onchange_tab1_l.emit(self.get_frame(frame_L, False))
            self.onchange_tab1_r.emit(self.get_frame(frame_R, False))
            self.onchange_tab2_l.emit(self.get_frame(dst_left, True))
            self.onchange_tab2_r.emit(self.get_frame(dst_right, True))
            self.onchange_tab3.emit(self.get_frame(filteredImg, True))


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = App()
    a = app.exec_()
    sys.exit(a)