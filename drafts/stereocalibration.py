import os
import cv2
import numpy as np
from stereovision.calibration import StereoCalibrator
from stereovision.calibration import StereoCalibration
from stereovision.exceptions import ChessboardNotFoundError


image_size = (640, 480)
path = 'D:\learn\8 sem\diplom\diplom'
path2 = 'resources\pictures\calibration_photos1'
right = 'photo_right'
left = 'photo_left'
ext = '.jpg'

# Chessboard parameters
rows = 6
columns = 8
square_size = 3

def calibration():
    calibrator = StereoCalibrator(rows, columns, square_size, image_size)

    print('Start processing')
    for i in range(1, 16):
        im_path_left = os.path.join(path,path2, left) + str(i) + ext
        im_path_right = os.path.join(path,path2, right) + str(i) + ext

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
    print ('End processing')

    calibration = calibrator.calibrate_cameras()
    calibration.export('D:\learn\8 sem\diplom\diplom\calib_result')
    print ('Calibration complete!')

if __name__ == '__main__':
    calibration()