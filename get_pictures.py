import cv2
import time
import os


left_cam = cv2.VideoCapture(1)
right_cam = cv2.VideoCapture(2)

pict_counter = 1
num_of_ph = 16
timer = 0

cv2.namedWindow('Left',cv2.WINDOW_AUTOSIZE)
cv2.namedWindow('Right',cv2.WINDOW_AUTOSIZE)


path_l = os.path.join('D:\learn\8 sem\diplom\diplom', 'resources\pictures\calibration_photos1\photo_left')
path_r = os.path.join('D:\learn\8 sem\diplom\diplom', 'resources\pictures\calibration_photos1\photo_right')

while True:
    f1, frame_L = left_cam.read()
    f2, frame_R = right_cam.read()

    time.sleep(1)
    timer += 1

    if timer == 5:
        cv2.imwrite(path_l + str(pict_counter) + '.jpg', frame_L)
        cv2.imwrite(path_r + str(pict_counter) + '.jpg', frame_R)
        print(str(pict_counter) + ' pair saved')
        pict_counter += 1
        timer = 0

    cv2.putText(frame_L, str(timer + 1), (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 0, 255), 3, cv2.LINE_AA)
    cv2.putText(frame_R, str(timer + 1), (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 0, 255), 3, cv2.LINE_AA)

    cv2.imshow('Left', frame_L)
    cv2.imshow('Right', frame_R)

    key = cv2.waitKey(1) & 0xFF
    if (key == ord("q")) | (pict_counter == num_of_ph):
        break

right_cam.release()
left_cam.release()
cv2.destroyAllWindows()