#Финальный файл
import cv2
import numpy as np
from imutils.perspective import four_point_transform
import imagehash
from PIL import Image
import os
import calibration_opencv


pedestrian_ref_0 = imagehash.average_hash(Image.open(os.path.join('D:\learn\8 sem\diplom\diplom','resources\pictures','road_signs\pedestrian_cross_ref_0.jpg')))
pedestrian_ref_90 = imagehash.average_hash(Image.open(os.path.join('D:\learn\8 sem\diplom\diplom','resources\pictures','road_signs\pedestrian_cross_ref_90.jpg')))
pedestrian_ref_180 = imagehash.average_hash(Image.open(os.path.join('D:\learn\8 sem\diplom\diplom','resources\pictures','road_signs\pedestrian_cross_ref_180.jpg')))
pedestrian_ref_270 = imagehash.average_hash(Image.open(os.path.join('D:\learn\8 sem\diplom\diplom','resources\pictures','road_signs\pedestrian_cross_ref_270.jpg')))
stop = imagehash.average_hash(Image.open(os.path.join('D:\learn\8 sem\diplom\diplom','resources\pictures','road_signs\stop.jpg')))


def hamming_distance(chaine1, chaine2):
    return sum(c1 != c2 for c1, c2 in zip(chaine1, chaine2))

def hash_compare_circle(fragment):
    segment = Image.fromarray(fragment)
    frag_hash = imagehash.average_hash(segment)
    angle0 = hamming_distance(str(frag_hash), str(stop))
    if angle0 <= 8:
        return True
    else:
        return False


def hash_compare_square(fragment):
    segment = Image.fromarray(fragment)
    frag_hash = imagehash.average_hash(segment)
    angle0 = hamming_distance(str(frag_hash), str(pedestrian_ref_0))
    angle1 = hamming_distance(str(frag_hash), str(pedestrian_ref_90))
    angle2 = hamming_distance(str(frag_hash), str(pedestrian_ref_180))
    angle3 = hamming_distance(str(frag_hash), str(pedestrian_ref_270))
    # print(angle0, angle1, angle2, angle3)
    if angle0 <= 10 or angle1 <= 10 or angle2 <= 10 or angle3 <= 10:
        return True
    else:
        return False


def search_signs(image):
    boxes = []
    circles = []

    # нижний и верхний цветовой порог
    hsv_min = np.array((26, 117, 82), np.uint8)
    hsv_max = np.array((122, 255, 255), np.uint8)

    # переводим в цветовое пространство hsv и размываем
    hsv = cv2.cvtColor(image,cv2.COLOR_BGR2HSV)
    blur = cv2.medianBlur(hsv, 3)
    # cv2.imshow('blur',hsv_blur)

    # выделяем цветовой диапазон нужных нам знаков
    thresh = cv2.inRange(blur, hsv_min, hsv_max)
    cv2.imshow('thresh', thresh)

    mask = cv2.erode(thresh, None, iterations=3)
    mask = cv2.dilate(mask, None, iterations=5)
    #cv2.imshow('mask', mask)

    #выделяем контуры детектором Канни
    canny_detector = cv2.Canny(mask, 25, 200,apertureSize=3)
    #cv2.imshow('canny',canny_detector)

    #находим контуры на изображении после детектора
    contours = cv2.findContours(canny_detector, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[0]

    # проходимся по списку найденных контуров
    # и пытаемся вписать в них прямоугольник
    # если это удаётся, считаем площадь прямоугольника
    # если она является максимальной на текущий момент
    # то сохраняем координаты вершин фигуры

    if len(contours) > 0:
        # проход по списку контуров, в поиске квадратных знаков
        for cnt in contours:
            rect = cv2.minAreaRect(cnt) # объект найденного квадрата
            width, height = rect[1]
            area = width * height # площадь
            if area > 2000: # ограничение по площади
                box = cv2.boxPoints(rect)
                box = np.int0(box)
                fragment = four_point_transform(image, [box][0]) # вырезаем по координатам
                flag = hash_compare_square(fragment) # проверяем, знак ли, возвращает True или False
                cv2.imshow('s', fragment)
                if flag:
                    boxes.append(rect) # если знак, то в список найденных

        for cnt in contours:
            # проход по списку контуров, в поиске круглых знаков
            center, radius = cv2.minEnclosingCircle(cnt)# вписываем круг
            centerx = int(center[0])
            centery = int(center[1])
            radius = int(radius)
            area = 3.14 * radius * radius
            if area > 500: # ограничение по площади
                rect = cv2.minAreaRect(cnt) # вписываем туда квадрат
                box = cv2.boxPoints(rect) # объект квадрата
                box = np.int0(box)
                fragment = four_point_transform(image, [box][0]) # вырезаем квадратную область
                flag = hash_compare_circle(fragment) # проверяем, знак ли, возвращает True или False

                if flag:
                    circles.append([centerx, centery, radius]) # если знак, то в список найденных

    return boxes, circles


def main():
    left_cam = cv2.VideoCapture(1)
    right_cam = cv2.VideoCapture(2)

    if not os.path.exists(os.path.join('D:\learn\8 sem\diplom\diplom','resources\configs\left_cam', 'left_cam_m.npy')):
        calibration_opencv.calibrate_cameras()
    print('Load settings')
    cm_left = np.load(os.path.join('D:\learn\8 sem\diplom\diplom', 'resources\configs\left_cam', 'left_cam_m.npy'))
    dc_left = np.load(os.path.join('D:\learn\8 sem\diplom\diplom', 'resources\configs\left_cam', 'left_cam_dist.npy'))
    cm_right = np.load(
        os.path.join('D:\learn\8 sem\diplom\diplom', 'resources\configs', 'right_cam', 'right_cam_m.npy'))
    dc_right = np.load(
        os.path.join('D:\learn\8 sem\diplom\diplom', 'resources\configs', 'right_cam', 'right_cam_dist.npy'))
    r = np.load(os.path.join('D:\learn\8 sem\diplom\diplom', 'resources\configs', 'r_mat.npy'))
    t = np.load(os.path.join('D:\learn\8 sem\diplom\diplom', 'resources\configs', 't_mat.npy'))
    e = np.load(os.path.join('D:\learn\8 sem\diplom\diplom', 'resources\configs', 'e_mat.npy'))
    f = np.load(os.path.join('D:\learn\8 sem\diplom\diplom', 'resources\configs', 'f_mat.npy'))

    while True:
        box_center, circle_center = None, None
        max_rect_left, max_rect_right = None, None
        max_circle_left, max_circle_right = None, None
        circle, box = False, False

        f1, frame_L = left_cam.read()
        f2, frame_R = right_cam.read()

        if not f1 or not f2:
            print('Failed to load stream')
            break

        box_left, circle_left = search_signs(frame_L)
        box_right, circle_right = search_signs(frame_R)
        print(len(box_left),' ',len(box_right))

        # если найдены квадраты в левом и правом
        if len(box_left) > 0 and len(box_right) and len(box_left) == len(box_right):
            print('Найдены некоторые квадраты!')
            max_area_left = 0
            max_area_right = 0
            for left, right in zip(box_left, box_right):# идём по списку левых квадратов и правых
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
            cv2.drawContours(frame_L,[box_l], 0, (0, 255, 0), 2) # рисуем квадрат на левом
            center_left = max_rect_left[0]

            # координаты правого
            box_r = cv2.boxPoints(max_rect_right)
            box_r = np.int0(box_r)
            cv2.drawContours(frame_R, [box_r], 0, (0, 255, 0), 2)  # рисуем квадрат на правом
            center_right = max_rect_right[0]

            center_x = int((center_left[0] + center_right[0])) // 2 # совмещаем координаты
            center_y = int((center_left[1] + center_right[1])) // 2 # найденных квадратов

            box_center = [center_x, center_y] # точка для определения расстояния
            box = True

        # если найдены круги в левом и правом
        if circle_left and circle_right and len(circle_left) == len(circle_right):
            print('Найдены некоторые круги!')
            max_area_left = 0
            max_area_right = 0
            for left, right in zip(circle_left, circle_right): # идём по списку левых и правых кружков
                left_r = left[2]
                area = 3.14 * left_r * left_r
                if area > max_area_left:
                    max_circle_left = left

                right_r = right[2]
                area = 3.14 * right_r * right_r
                if area > max_area_right:
                    max_circle_right = left

            # левый
            center_left = [max_circle_left[0], max_circle_left[1]]  # центр
            rad_left = max_circle_left[2]  # радиус
            cv2.circle(frame_L, center_left, rad_left, (0, 0, 255), 2)  # рисуем кружок на левом

            # правый
            center_right = [max_circle_right[0], max_circle_right[1]]  # центр
            rad_right = max_circle_right[2]  # радиус
            cv2.circle(frame_R, center_right, rad_right, (0, 0, 255), 2)  # рисуем кружок на левом

            center_x = (center_left[0] + center_right[0]) // 2  # совмещаем координаты
            center_y = (center_left[1] + center_right[1]) // 2  # найденных кружков

            circle_center = [center_x, center_y] # точка для определения расстояния
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
        matcher = cv2.StereoSGBM_create(minDisparity=31,
                                             numDisparities=112,
                                             blockSize=8,
                                             P1=8 * 3 * 8 ** 2,
                                             # wsize default 3; 5; 7 for SGBM reduced size image; 15 for SGBM full size image (1300px and above); 5 Works nicely
                                             P2=32 * 3 * 8 ** 2,
                                             disp12MaxDiff=60,
                                             uniquenessRatio=10,
                                             speckleWindowSize=10,
                                             speckleRange=32,
                                             mode=False)
        disparity = matcher.compute(dst_left, dst_right).astype(np.float32) / 16
        disparity = cv2.normalize(disparity, disparity, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
        #disparity = cv2.medianBlur(disparity, 5)

        if box:
            x1,y1 = max_rect_left[0]
            x2,y2 = max_rect_right[0]

            dot = disparity[box_center[0]][box_center[1]]
            distance = round(0.004 * 0.072 / 0.0495 * dot,5)  # (focal_l * base) / (size_of_pix * dispar)
            cv2.putText(frame_L, str(distance), (int(x1), int(y1)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), lineType=cv2.LINE_AA)
            cv2.putText(frame_R, str(distance), (int(x2), int(y2)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), lineType=cv2.LINE_AA)
        elif circle:
            x1, y1 = max_circle_left[0], max_circle_left[1]
            x2, y2 = max_circle_right[0], max_circle_right[1]

            dot = disparity[circle_center[0]][circle_center[1]]
            distance = round(0.004 * 0.072 / 0.0495 * dot, 3)  # (focal_l * base) / (size_of_pix * dispar)
            cv2.putText(frame_L, str(distance), (int(x1), int(y1)), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0),
                        lineType=cv2.LINE_AA)
            cv2.putText(frame_R, str(distance), (int(x2), int(y2)), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0),
                        lineType=cv2.LINE_AA)

        cv2.imshow('Left', frame_L)
        cv2.imshow('Right', frame_R)
        cv2.imshow('Disparity', disparity)

        if cv2.waitKey(1) & 0xFF is ord('q'):
            left_cam.release()
            right_cam.release()
            cv2.destroyAllWindows()
            break


if __name__ == '__main__':
    main()
