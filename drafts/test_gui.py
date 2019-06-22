import cv2
import numpy as np
import sys
from stereovision.calibration import StereoCalibrator
from stereovision.exceptions import ChessboardNotFoundError
from imutils.perspective import four_point_transform
import imagehash
from PIL import Image
import time
import os.path
from stereovision.calibration import StereoCalibration
from PyQt5.QtCore import QThread, pyqtSignal, pyqtSlot, Qt
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QWidget, QLabel, QApplication, QHBoxLayout, \
    QVBoxLayout, QMainWindow, QTabWidget, QLabel, QPushButton

pedestrian_ref_0 = imagehash.average_hash(Image.open('resources/pictures/road_signs/pedestrian_cross_ref_0.jpg'))
pedestrian_ref_90 = imagehash.average_hash(Image.open('resources/pictures/road_signs/pedestrian_cross_ref_90.jpg'))
pedestrian_ref_180 = imagehash.average_hash(Image.open('resources/pictures/road_signs/pedestrian_cross_ref_180.jpg'))
pedestrian_ref_270 = imagehash.average_hash(Image.open('resources/pictures/road_signs/pedestrian_cross_ref_270.jpg'))
dont_stop = imagehash.average_hash(Image.open('resources/pictures/road_signs/dont_stop_ref.jpg'))


class App(QMainWindow):
    def __init__(self):
        super().__init__()
        self.table_widget = MainWin()
        self.setCentralWidget(self.table_widget)
        self.setWindowTitle('Test_win')
        self.setGeometry(300, 50, 1024, 768)


class MainWin(QWidget):
    def __init__(self):
        super().__init__()
        self.im1 = cv2.imread('left1.jpg')
        self.im2 = cv2.imread('right.jpg')
        self.im3 = cv2.imread('van_dam.jpg')
        im4 = QImage('please-stand-by.jpg').scaled(640, 480, Qt.KeepAspectRatio)
        self.start_image = QPixmap.fromImage(im4)

        self.range_detector = RangeDetector(self)

        self.Layout = QVBoxLayout()

        self.tabs = QTabWidget()
        self.first_tab = QWidget()
        self.second_tab = QWidget()

        self.tabs.addTab(self.first_tab, 'Первая вкладка')
        self.tabs.addTab(self.second_tab, 'Вторая вкладка')

        self.left_lbl = QLabel()
        self.right_lbl = QLabel()

        self.central_lbl = QLabel()

        self.start = QPushButton('Start')
        self.start.clicked.connect(self.set_pict)
        self.stop = QPushButton('Stop')
        self.stop.clicked.connect(self.clear)

        self.first_tab_ui()
        self.second_tab_ui()

        self.left_lbl.setPixmap(self.start_image)
        self.right_lbl.setPixmap(self.start_image)
        self.central_lbl.setPixmap(self.start_image)

        self.Layout.addWidget(self.tabs)
        self.setLayout(self.Layout)

    def first_tab_ui(self):
        self.tabs.setTabText(0, '1')

        vbox = QVBoxLayout()
        hbox = QHBoxLayout()
        buttons = QHBoxLayout()

        hbox.addWidget(self.left_lbl, alignment=Qt.AlignCenter)
        hbox.addWidget(self.right_lbl, alignment=Qt.AlignCenter)

        buttons.addWidget(self.start, alignment=Qt.AlignCenter)
        buttons.addWidget(self.stop, alignment=Qt.AlignCenter)

        vbox.addLayout(hbox)
        vbox.addLayout(buttons)

        self.first_tab.setLayout(vbox)

    def second_tab_ui(self):
        self.tabs.setTabText(1, '2')
        vbox = QVBoxLayout()

        vbox.addWidget(self.central_lbl, alignment=Qt.AlignCenter)

        self.second_tab.setLayout(vbox)

    def get_qimage(self,im):
        height, width, channel = im.shape
        bytes = channel * width
        convert_image = QImage(im.data, width, height, bytes, QImage.Format_RGB888).scaled(640, 480, Qt.KeepAspectRatio)
        result_image = convert_image.rgbSwapped()

        return result_image

    @pyqtSlot()
    def set_pict(self):
        self.left_lbl.setPixmap(QPixmap.fromImage(self.get_qimage(self.im1)))
        self.left_lbl.setPixmap(QPixmap.fromImage(self.get_qimage(self.im2)))
        self.left_lbl.setPixmap(QPixmap.fromImage(self.get_qimage(self.im3)))

        self.range_detector.start()
        self.range_detector.change_left.connect(self.set_left)
        self.range_detector.change_right.connect(self.set_right)
        # self.range_detector.change_center.connect(self.set_center)

    @pyqtSlot()
    def clear(self):
        self.range_detector.stop()
        self.range_detector.quit()
        self.range_detector.wait()
        self.left_lbl.setPixmap(self.start_image)
        self.right_lbl.setPixmap(self.start_image)
        self.central_lbl.setPixmap(self.start_image)

    @pyqtSlot(QImage)
    def set_left(self, image):
        self.left_lbl.setPixmap(QPixmap.fromImage(image))

    @pyqtSlot(QImage)
    def set_right(self, image):
        self.right_lbl.setPixmap(QPixmap.fromImage(image))

    # @pyqtSlot(QImage)
    # def set_center(self, image):
    #     self.central_lbl.setPixmap(QPixmap.fromImage(image))


class RangeDetector(QThread):
    change_left = pyqtSignal(QImage)
    change_right = pyqtSignal(QImage)
    change_center = pyqtSignal(QImage)

    def __init__(self, parent):
        super(QThread,self).__init__(parent)
        self.left_cam = cv2.VideoCapture(1)
        self.right_cam = cv2.VideoCapture(2)
        # print('Поток запущен')
        self.lets_get_it_started()

    def stop(self):
        self.running = False

    def hamming_distance(self, chaine1, chaine2):
        return sum(c1 != c2 for c1, c2 in zip(chaine1, chaine2))

    def hash_compare_circle(self, fragment):
        segment = Image.fromarray(fragment)
        frag_hash = imagehash.average_hash(segment)
        angle0 = self.hamming_distance(str(frag_hash), str(dont_stop))
        # print(angle0)
        if angle0 <= 8:
            return True
        return False

    def hash_compare_square(self, fragment):
        segment = Image.fromarray(fragment)
        frag_hash = imagehash.average_hash(segment)
        angle0 = self.hamming_distance(str(frag_hash), str(pedestrian_ref_0))
        angle1 = self.hamming_distance(str(frag_hash), str(pedestrian_ref_90))
        angle2 = self.hamming_distance(str(frag_hash), str(pedestrian_ref_180))
        angle3 = self.hamming_distance(str(frag_hash), str(pedestrian_ref_270))
        # print(angle0, angle1, angle2, angle3)
        if angle0 <= 10 or angle1 <= 10 or angle2 <= 10 or angle3 <= 10:
            return True
        return False

    def calibrate(self):
        rows = 6
        columns = 8
        square_size = 3
        image_size = (640, 480)

        path = 'resources/pictures/calibration_photos/'
        right = 'photo_right'
        left = 'photo_left'
        ext = '.jpg'

        calibrator = StereoCalibrator(rows, columns, square_size, image_size)

        #print('Start processing')
        for i in range(1, 16):
            im_path_left = os.path.join(path, left) + str(i) + ext
            im_path_right = os.path.join(path, right) + str(i) + ext

            if os.path.exists(im_path_left) and os.path.exists(im_path_right):
                img_left = cv2.imread(im_path_left)
                img_right = cv2.imread(im_path_right)
            else:
                print(str(i) + ' step failed. Wrong path.')
                continue

            try:
                calibrator._get_corners(img_left)
                calibrator._get_corners(img_right)
            except ChessboardNotFoundError as error:
                print(error)
                print("Pair No " + str(i) + " ignored")
            else:
                calibrator.add_corners((img_left, img_right), True)
        print('End processing')
        return

    def search_signs(self,image):
        boxes = []
        circles = []

        # нижний и верхний цветовой порог
        hsv_min = np.array((26, 117, 82), np.uint8)
        hsv_max = np.array((122, 255, 255), np.uint8)
        # hsv_min = np.array((19, 93, 91), np.uint8)
        # hsv_max = np.array((220, 255, 255), np.uint8)

        # переводим в цветовое пространство hsv и размываем
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        blur = cv2.medianBlur(hsv, 3)
        # cv2.imshow('blur',hsv_blur)

        # выделяем цветовой диапазон нужных нам знаков
        thresh = cv2.inRange(blur, hsv_min, hsv_max)
        #cv2.imshow('thresh', thresh)

        mask = cv2.erode(thresh, None, iterations=3)
        mask = cv2.dilate(mask, None, iterations=5)
        #cv2.imshow('mask', mask)

        # выделяем контуры детектором Канни
        canny_detector = cv2.Canny(mask, 25, 200, apertureSize=3)
        #cv2.imshow('canny', canny_detector)

        # находим контуры на изображении после детектора
        contours = cv2.findContours(canny_detector, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE, )[1]

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
                    # cv2.drawContours(image, [box], 0, (0, 255, 0), 2)
                    fragment = four_point_transform(image, [box][0])  # вырезаем по координатам
                    flag = self.hash_compare_square(fragment)  # проверяем, знак ли, возвращает True или False
                    #cv2.imshow('s', fragment)
                    if flag:
                        # cv2.drawContours(image, [box], 0, (0, 255, 0), 2)  # рисуем квадрат на левом
                        boxes.append(rect)  # если знак, то в список найденных

            for cnt in contours:
                # проход по списку контуров, в поиске круглых знаков
                center, radius = cv2.minEnclosingCircle(cnt)  # вписываем круг
                centerx = int(center[0])
                centery = int(center[1])
                radius = int(radius)
                area = 3.14 * radius * radius
                if area > 500:  # ограничение по площади
                    rect = cv2.minAreaRect(cnt)  # вписываем туда квадрат
                    box = cv2.boxPoints(rect)  # объект квадрата
                    box = np.int0(box)
                    fragment = four_point_transform(image, [box][0])  # вырезаем квадратную область
                    flag = self.hash_compare_circle(fragment)  # проверяем, знак ли, возвращает True или False

                    if flag:
                        circles.append([centerx, centery, radius])  # если знак, то в список найденных

        return boxes, circles

    def paint(self, box_left, box_right, circle_left, circle_right, frame_L, frame_R):
        box_centers = []
        circle_centers = []

        #print(len(box_left), ' ', len(box_right))
        if len(box_left) > 0 and len(box_right) and len(box_left) == len(
                box_right):  # если найдены квадраты в левом и правом
            #print('Найдены некоторые квадраты!')
            for left, right in zip(box_left, box_right):  # идём по списку левых квадратов и правых
                # координаты левого
                box_l = cv2.boxPoints(left)
                box_l = np.int0(box_l)
                cv2.drawContours(frame_L, [box_l], 0, (0, 255, 0), 2)  # рисуем квадрат на левом
                center_left = left[0]

                # координаты правого
                box_r = cv2.boxPoints(right)
                box_r = np.int0(box_r)
                cv2.drawContours(frame_R, [box_r], 0, (0, 255, 0), 2)  # рисуем квадрат на правом
                center_right = right[0]

                center_x = int((center_left[0] + center_right[0])) // 2  # совмещаем координаты
                center_y = int((center_left[1] + center_right[1])) // 2  # найденных квадратов

                box_centers.append([center_x, center_y])  # точка для определения расстояния

        if len(circle_left) > 0 and len(circle_right) > 0 and len(circle_left) == len(
                circle_right):  # если найдены круги в левом и правом
            #print('Найдены некоторые круги!')
            for left, right in zip(circle_left, circle_right):  # идём по списку левых и правых кружков
                # левый
                center_left = [left[0], left[1]]  # центр
                rad_left = left[2]  # радиус
                cv2.circle(frame_L, center_left, rad_left, (0, 0, 255), 2)  # рисуем кружок на левом

                # правый
                center_right = [right[0], right[1]]  # центр
                rad_right = right[2]  # радиус
                cv2.circle(frame_R, center_right, rad_right, (0, 0, 255), 2)  # рисуем кружок на левом

                center_x = (center_left[0] + center_right[0]) // 2  # совмещаем координаты
                center_y = (center_left[1] + center_right[1]) // 2  # найденных кружков

                circle_centers.append([center_x, center_y])  # точка для определения расстояния
        return box_centers, circle_centers

    def create_disparity(self, frame_L, frame_R):
        rectified_pair = self.calibration.rectify((frame_L, frame_R))
        stereo = cv2.StereoSGBM_create(90, 32, 2, 600, 2400, 20, 16, 1, 100, 20, False)
        disparity = stereo.compute(rectified_pair[0], rectified_pair[1])
        disparity = cv2.normalize(disparity, disparity, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
        disparity = cv2.medianBlur(disparity, 5)
        return disparity

    def lets_get_it_started(self):
        self.running = True
        # check files, if not - calibrate stereocamera
        if not os.path.exists('calib_result'):
            self.calibrate()
        print('Load settings')

        # load data
        self.calibration = StereoCalibration(input_folder='calib_result')

        self.distances = []
        f1, frame_L = self.left_cam.read()
        f2, frame_R = self.right_cam.read()

        while self.running:
            if f1 and f2:
                tic = round(time.time(), 3)
                box_left, circle_left = self.search_signs(frame_L)
                box_right, circle_right = self.search_signs(frame_R)

                box_centers, circle_centers = self.paint(box_left, box_right, circle_left, circle_right, frame_L, frame_R)

                toc = round(time.time(), 3)
                cv2.putText(frame_L, str(toc - tic), (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0),
                            lineType=cv2.LINE_AA)
                disparity = self.create_disparity(frame_L, frame_R)

                for center in box_centers:
                    dot = disparity[center[0]][center[1]]
                    distance = 0.004 * 0.072 / 0.0495 * dot  # (focal_l * base) / (size_of_pix * dispar)
                    self.distances.append(distance)

                left_im = self.get_qimage(frame_L, True)
                right_im = self.get_qimage(frame_R, True)
                # disparity_map = self.get_qimage(disparity, False)

                self.change_left.emit(left_im)
                self.change_right.emit(right_im)
                # self.change_center.emit(disparity_map)

    def get_qimage(self, im, isdisp):
        if isdisp:
            height, width, channel = im.shape
            bytes = channel * width
            convert_image = QImage(im.data, width, height, bytes, QImage.Format_RGB888)\
                .scaled(640, 480, Qt.KeepAspectRatio)
            result_image = convert_image.rgbSwapped()
        else: # не работает
            height, width = im.shape
            bytes = 1 * width
            convert_image = QImage(im.data, width, height, bytes, QImage.Format_RGB888)\
                .scaled(640, 480, Qt.KeepAspectRatio)
            result_image = convert_image.rgbSwapped()
        return result_image


def main():
    app = QApplication(sys.argv)
    mw = App()
    mw.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()