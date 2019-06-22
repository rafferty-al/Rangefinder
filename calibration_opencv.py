# лучше сделать запись файл и загрузку параметров после калибровки
import cv2
import os
import numpy as np


path = os.path.join('D:\learn\8 sem\diplom\diplom','resources\pictures\calibration_photos1')

right = 'photo_right'
left = 'photo_left'
ext = '.jpg'


def calibrate_cameras():

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    obj_points = np.zeros((6 * 8, 3), np.float32)
    obj_points[:, :2] = np.mgrid[0:8, 0:6].T.reshape(-1, 2)

    # Arrays to store object points and image points from all the images.
    object_points = [] # координаты в 3D пространстве
    image_points_left = [] # 2D координаты на левом изображении
    image_points_right = [] # 2D координаты на правом изображении

    print('-Start processing cameras')
    for i in range(1, 16):
        im_path_left = os.path.join(path, left) + str(i) + ext
        im_path_right = os.path.join(path, right) + str(i) + ext

        if os.path.exists(im_path_left) and os.path.exists(im_path_right):
            img_left = cv2.imread(im_path_left)
            img_right = cv2.imread(im_path_right)
        else:
            print(str(i) + ' step failed. Wrong path.')
            continue

        gray_left = cv2.cvtColor(img_left, cv2.COLOR_BGR2GRAY)
        gray_right = cv2.cvtColor(img_right, cv2.COLOR_BGR2GRAY)

        # Ищем углы на шахматной доске
        ret_l, corners_left = cv2.findChessboardCorners(gray_left, (8, 6), None)
        ret_r, corners_right = cv2.findChessboardCorners(gray_right, (8, 6), None)

        # Если углы найдены, добавляем их пространственные координаты и точки на изображении
        if ret_l and ret_r:
            object_points.append(obj_points)

            corner_l=cv2.cornerSubPix(gray_left,corners_left, (11,11), (-1,-1), criteria)
            corner_r=cv2.cornerSubPix(gray_right,corners_right, (11,11), (-1,-1), criteria)

            image_points_left.append(corner_l)
            image_points_right.append(corner_r)

            # Рисуем углы на фотографиях
            cv2.drawChessboardCorners(img_left, (8,6), corner_l, ret_l)
            cv2.drawChessboardCorners(img_right, (8,6), corner_r, ret_r)

            cv2.imshow('img_left', img_left)
            cv2.imshow('img_right', img_right)
            cv2.waitKey(500)

    cv2.destroyAllWindows()
    print('-Processing completed!')

    print('-Start calibrate cameras')
    #получение матрицы камеры и коэффициентов искажений
    ret_l, mtx_l, dist_l, rvecs_l, tvecs_l = cv2.calibrateCamera(object_points, image_points_left, (640, 480), None, None)
    ret_r, mtx_r, dist_r, rvecs_r, tvecs_r = cv2.calibrateCamera(object_points, image_points_right, (640, 480), None, None)
    retval, cam_mtrx_left, dst_coeff_left, cam_mtrx_right, dst_coeff_right, r, t, e, f = \
        cv2.stereoCalibrate(object_points,image_points_left,image_points_right,mtx_l,dist_l,mtx_r, dist_r,(640,480))
    print('-Calibrated completed')

    np.save(os.path.join('D:\learn\8 sem\diplom\diplom','resources\configs\left_cam', 'left_cam_m') , cam_mtrx_left)
    np.save(os.path.join('D:\learn\8 sem\diplom\diplom','resources\configs\left_cam', 'left_cam_dist'), dst_coeff_left)
    np.save(os.path.join('D:\learn\8 sem\diplom\diplom','resources\configs','right_cam', 'right_cam_m'), cam_mtrx_right)
    np.save(os.path.join('D:\learn\8 sem\diplom\diplom','resources\configs','right_cam', 'right_cam_dist'), dst_coeff_right)
    np.save(os.path.join('D:\learn\8 sem\diplom\diplom','resources\configs', 'r_mat'), r)
    np.save(os.path.join('D:\learn\8 sem\diplom\diplom','resources\configs', 't_mat'), t)
    np.save(os.path.join('D:\learn\8 sem\diplom\diplom','resources\configs', 'e_mat'), e)
    np.save(os.path.join('D:\learn\8 sem\diplom\diplom','resources\configs', 'f_mat'), f)

    #return cam_mtrx_left, dst_coeff_left, cam_mtrx_right, dst_coeff_right,  r, t, e, f


if __name__ == '__main__':
    calibrate_cameras()