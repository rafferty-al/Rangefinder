#time = 0.02-0.03 sec
import cv2
import numpy as np
from imutils.perspective import four_point_transform
import imagehash
from PIL import Image
import time
import os


pedestrian_ref_0 = imagehash.average_hash(Image.open(os.path.join('D:\diplom\diplom','resources\pictures','road_signs\pedestrian_cross_ref_0.jpg')))
pedestrian_ref_90 = imagehash.average_hash(Image.open(os.path.join('D:\diplom\diplom','resources\pictures','road_signs\pedestrian_cross_ref_90.jpg')))
pedestrian_ref_180 = imagehash.average_hash(Image.open(os.path.join('D:\diplom\diplom','resources\pictures','road_signs\pedestrian_cross_ref_180.jpg')))
pedestrian_ref_270 = imagehash.average_hash(Image.open(os.path.join('D:\diplom\diplom','resources\pictures','road_signs\pedestrian_cross_ref_270.jpg')))
brick_0 = imagehash.average_hash(Image.open(os.path.join('D:\diplom\diplom', 'resources\pictures', 'road_signs','brick.jpg')))
brick_45 = imagehash.average_hash(Image.open(os.path.join('D:\diplom\diplom', 'resources\pictures', 'road_signs','brick_45.jpg')))
brick_90 = imagehash.average_hash(Image.open(os.path.join('D:\diplom\diplom', 'resources\pictures', 'road_signs','brick_90.jpg')))
brick_135 = imagehash.average_hash(Image.open(os.path.join('D:\diplom\diplom', 'resources\pictures', 'road_signs','brick_135.jpg')))

def hamming_distance(chaine1, chaine2):
    return sum(c1 != c2 for c1, c2 in zip(chaine1, chaine2))

def hash_compare_circle(fragment):
    segment = Image.fromarray(fragment)
    frag_hash = imagehash.average_hash(segment)
    angle0 = hamming_distance(str(frag_hash), str(brick_0))
    angle1 = hamming_distance(str(frag_hash), str(brick_45))
    angle2 = hamming_distance(str(frag_hash), str(brick_90))
    angle3 = hamming_distance(str(frag_hash), str(brick_135))
    print(angle0,angle1,angle2,angle3)
    if angle0 <= 11 or angle1 <= 11 or angle2 <= 11 or angle3 <= 11:
        return True
    else:
        return False

def hash_compare(fragment):
    segment = Image.fromarray(fragment)
    frag_hash = imagehash.average_hash(segment)
    angle0 = hamming_distance(str(frag_hash), str(pedestrian_ref_0))
    angle1 = hamming_distance(str(frag_hash), str(pedestrian_ref_90))
    angle2 = hamming_distance(str(frag_hash), str(pedestrian_ref_180))
    angle3 = hamming_distance(str(frag_hash), str(pedestrian_ref_270))
    if angle0 <= 0 or angle1 <= 0 or angle2 <= 0 or angle3 <= 0:
        return True
    else:
        return False


def search_square_sign(image):
    # нижний и верхний цветовой порог
    hsv_min = np.array((0, 92, 86), np.uint8)
    hsv_max = np.array((255, 255, 255), np.uint8)
    #hsv_min = np.array((19, 93, 91), np.uint8)
    #hsv_max = np.array((220, 255, 255), np.uint8)

    # переводим в цветовое пространство hsv и размываем
    hsv = cv2.cvtColor(image,cv2.COLOR_BGR2HSV)
    blur = cv2.medianBlur(hsv, 3)
    # cv2.imshow('blur',hsv_blur)

    # выделяем цветовой диапазон нужных нам знаков
    thresh = cv2.inRange(blur, hsv_min, hsv_max)
    cv2.imshow('thresh', thresh)

    # mask = cv2.erode(thresh, None, iterations=3)
    # mask = cv2.dilate(mask, None, iterations=5)
    # cv2.imshow('mask', mask)

    #выделяем контуры детектором Канни
    canny_detector = cv2.Canny(thresh, 25, 200,apertureSize=3)
    cv2.imshow('canny',canny_detector)

    #находим контуры на изображении после детектора
    contours = cv2.findContours(canny_detector, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE,)[0]

    # проходимся по списку найденных контуров
    # и пытаемся вписать в них прямоугольник
    # если это удаётся, считаем площадь прямоугольника
    # если она является максимальной на текущий момент
    # то сохраняем координаты вершин фигуры

    if len(contours) > 0:
        for cnt in contours:
            # rect = cv2.minAreaRect(cnt)
            # width, height = rect[1]
            # area = width * height
            # if area > 8000:
            #     box = cv2.boxPoints(rect)
            #     box = np.int0(box)
            #     cv2.drawContours(image, [box], 0, (0, 255, 0), 2)
            #     fragment = four_point_transform(image, [box][0])
            #     flag = hash_compare(fragment)
            #
            #     if flag:
            #         coord = rect[0]
            #         return box, coord
            center, radius = cv2.minEnclosingCircle(cnt)  # вписываем круг
            centerx = int(center[0])
            centery = int(center[1])
            radius = int(radius)
            area = 3.14 * radius * radius
            if area > 4000:  # ограничение по площади

                rect = cv2.minAreaRect(cnt)  # вписываем туда квадрат
                box = cv2.boxPoints(rect)  # объект квадрата
                box = np.int0(box)
                fragment = four_point_transform(image, [box][0])  # вырезаем квадратную область
                cv2.imshow('s', fragment)
                flag = hash_compare_circle(fragment)  # проверяем, знак ли, возвращает True или False

                if flag:
                    cv2.circle(image, (centerx, centery), radius, (0, 255, 255), 2)

def main():
    # capture = cv2.VideoCapture(1)

    while True:
        # status, frame = capture.read()

        # if not status:
        #     print('Failed to load stream')
        #     break

        # frame = cv2.imread(os.path.join('D:\diplom\diplom','resources\pictures','test\left_61cm.jpg'))
        frame = cv2.imread(os.path.join('D:\diplom\diplom','resources\pictures','test1\left_62cm.jpg'))

        tic = round(time.time(),3)
        search_square_sign(frame)
        toc = round(time.time(),3)
        # cv2.putText(frame,str(toc - tic), (50, 50),cv2.FONT_HERSHEY_COMPLEX, 2,(0,0,255), lineType=cv2.LINE_AA)
        cv2.imshow('Frame', frame)


        if cv2.waitKey(1) & 0xFF is ord('q'):
            # capture.release()
            cv2.destroyAllWindows()
            break


if __name__ == '__main__':
    main()