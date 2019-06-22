import cv2
import numpy as np
import os


def nothing(*arg):
    pass

cap = cv2.VideoCapture(1)

cv2.namedWindow("result",cv2.WINDOW_NORMAL)
cv2.namedWindow("settings",cv2.WINDOW_NORMAL)

# создаем 6 бегунков для настройки начального и конечного цвета фильтра
cv2.createTrackbar('minb', 'settings', 26, 255, nothing)
cv2.createTrackbar('ming', 'settings', 117, 255, nothing)
cv2.createTrackbar('minr', 'settings', 82, 255, nothing)
cv2.createTrackbar('maxb', 'settings', 122, 255, nothing)
cv2.createTrackbar('maxg', 'settings', 255, 255, nothing)
cv2.createTrackbar('maxr', 'settings', 255, 255, nothing)
#crange = [0, 0, 0, 0, 0, 0]


while True:
    #img = cv2.imread('van_dam.jpg')
    #flag, img = cap.read()
    img = cv2.imread(os.path.join('D:\diplom\diplom','resources\pictures','test\left_61cm.jpg'))
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    #cv2.imshow('hsv',hsv)

    # считываем значения бегунков
    minb = cv2.getTrackbarPos('minb', 'settings')
    ming = cv2.getTrackbarPos('ming', 'settings')
    minr = cv2.getTrackbarPos('minr', 'settings')
    maxb = cv2.getTrackbarPos('maxb', 'settings')
    maxg = cv2.getTrackbarPos('maxg', 'settings')
    maxr = cv2.getTrackbarPos('maxr', 'settings')

    #размытие изображения
    hsv = cv2.blur(hsv,(5,5))
    #cv2.imshow('blur',hsv)

    # формируем начальный и конечный цвет фильтра
    min = np.array((minb, ming, minr), np.uint8)
    max = np.array((maxb, maxg, maxr), np.uint8)

    # накладываем фильтр на кадр в модели HSV
    mask = cv2.inRange(hsv, min, max)
    cv2.imshow('mask', mask)

    #обработка бинаризованного изображения
    mask_erode = cv2.erode(mask,None,iterations=2)
    #cv2.imshow('Erode',mask_erode)
    mask_dilate = cv2.dilate(mask,None,iterations=4)
    #cv2.imshow('Dilate',mask_dilate)

    result = cv2.bitwise_and(img, img, mask = mask)
    cv2.imshow('result',result)

    ch = cv2.waitKey(5)
    if ch == 27:
        break

cap.release()
cv2.destroyAllWindows()
