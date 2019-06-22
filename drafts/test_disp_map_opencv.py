import cv2
import os
import numpy as np
import time

#print('Load settings...')
cm_left = np.load(os.path.join('D:\learn\8 sem\diplom\diplom','resources\configs\left_cam', 'left_cam_m.npy'))
dc_left = np.load(os.path.join('D:\learn\8 sem\diplom\diplom','resources\configs\left_cam', 'left_cam_dist.npy'))
cm_right = np.load(os.path.join('D:\learn\8 sem\diplom\diplom','resources\configs','right_cam', 'right_cam_m.npy'))
dc_right = np.load(os.path.join('D:\learn\8 sem\diplom\diplom','resources\configs','right_cam', 'right_cam_dist.npy'))
r = np.load(os.path.join('D:\learn\8 sem\diplom\diplom','resources\configs', 'r_mat.npy'))
t = np.load(os.path.join('D:\learn\8 sem\diplom\diplom','resources\configs', 't_mat.npy'))
e = np.load(os.path.join('D:\learn\8 sem\diplom\diplom','resources\configs', 'e_mat.npy'))
f = np.load(os.path.join('D:\learn\8 sem\diplom\diplom','resources\configs', 'f_mat.npy'))

left_cam = cv2.VideoCapture(1)
right_cam = cv2.VideoCapture(2)
cv2.namedWindow('disparity',cv2.WINDOW_GUI_NORMAL)
cv2.namedWindow('left',cv2.WINDOW_GUI_NORMAL)
cv2.namedWindow('right',cv2.WINDOW_GUI_NORMAL)
cv2.namedWindow('settings',cv2.WINDOW_GUI_NORMAL)

cv2.createTrackbar('minDisparity','settings',36,100,lambda x: x)
cv2.createTrackbar('numDisparities','settings',112,208,lambda x:x)
cv2.createTrackbar('blockSize','settings',8,20,lambda x:x)
cv2.createTrackbar('disp12MaxDiff','settings',42,60,lambda x:x)
cv2.createTrackbar('uniquenessRatio','settings',3,60,lambda x:x)
cv2.createTrackbar('speckleWindowSize','settings',50,250,lambda x:x)
cv2.createTrackbar('speckleRange','settings',50,250,lambda x:x)
while True:

    minDisparity = cv2.getTrackbarPos('minDisparity','settings')
    numDisparities = cv2.getTrackbarPos('numDisparities','settings')
    blockSize = cv2.getTrackbarPos('blockSize','settings')
    disp12MaxDiff = cv2.getTrackbarPos('disp12MaxDiff','settings')
    uniquenessRatio = cv2.getTrackbarPos('uniquenessRatio','settings')
    speckleWindowSize = cv2.getTrackbarPos('speckleWindowSize','settings')
    speckleRange = cv2.getTrackbarPos('speckleRange','settings')

    # print('Open photos')
    img_left = cv2.imread(os.path.join('D:\learn\8 sem\diplom\diplom','resources\pictures','test\left_61cm.jpg'))
    img_right = cv2.imread(os.path.join('D:\learn\8 sem\diplom\diplom','resources\pictures','test\lright_61cm.jpg'))
    # f1, img_left = left_cam.read()
    # f2, img_right = right_cam.read()

    h, w = img_left.shape[:2]

    #print('Start undistortion...')
    new_left_mtx, roi_l = cv2.getOptimalNewCameraMatrix(cm_left, dc_left, (w, h), 1, (w, h))
    new_right_mtx, roi_r = cv2.getOptimalNewCameraMatrix(cm_right, dc_right, (w, h), 1, (w, h))

    # print('Start stereoRectify...')
    # r_left, r_right, pm_left, pm_right, q, roi_left, roi_right = cv2.stereoRectify(cm_left, dc_left, cm_right, dc_right,
    #                                                                                (w, h), r, t)

    # undistort left
    mapx1, mapy1 = cv2.initUndistortRectifyMap(cm_left, dc_left, r, new_left_mtx, (w, h), 5)
    dst_left = cv2.remap(img_left, mapx1, mapy1, cv2.INTER_LINEAR)
    # x, y, w, h = roi_l
    # dst_left = dst_left[y:y + h, x:x + w]

    # undistort right
    mapx2, mapy2 = cv2.initUndistortRectifyMap(cm_right, dc_right, r, new_right_mtx, (w, h), 5)
    dst_right = cv2.remap(img_right, mapx2, mapy2, cv2.INTER_LINEAR)
    # x, y, w, h = roi_r
    # dst_right = dst_right[y:y + h, x:x + w]
    #print('Done!')

    dst_left = cv2.cvtColor(dst_left, cv2.COLOR_BGR2GRAY)
    dst_right = cv2.cvtColor(dst_right, cv2.COLOR_BGR2GRAY)

    window_size = 5

    lmbda = 80000
    sigma = 1.2
    visual_multiplier = 1.0

    tic = round(time.time(), 3)
    # print('create matchers')

    left_matcher = cv2.StereoSGBM_create(minDisparity=minDisparity,
                                 numDisparities=numDisparities,
                                 # max_disp has to be dividable by 16 f. E. HH 192, 256
                                 blockSize=blockSize,
                                 P1=8 * 3 * blockSize ** 2,
                                 # wsize default 3; 5; 7 for SGBM reduced size image; 15 for SGBM full size image (1300px and above); 5 Works nicely
                                 P2=32 * 3 * blockSize ** 2,
                                 disp12MaxDiff=disp12MaxDiff,
                                 uniquenessRatio=uniquenessRatio,
                                 speckleWindowSize=speckleWindowSize,
                                 speckleRange=speckleRange,
                                 mode=False)

    # right_matcher = cv2.ximgproc.createRightMatcher(left_matcher)
    #
    # # print('create filter')
    # wls_filter = cv2.ximgproc.createDisparityWLSFilter(matcher_left=left_matcher)
    # wls_filter.setLambda(lmbda)
    # wls_filter.setSigmaColor(sigma)

    # print('computing disparity...')
    try:
        displ = left_matcher.compute(dst_left, dst_right).astype(np.float32)/16
        # dispr = right_matcher.compute(dst_left, dst_right).astype(np.float32)/16
        # # displ = np.int16(displ)
        # # dispr = np.int16(dispr)
        # filteredImg = wls_filter.filter(displ, dst_left, None, dispr)  # important to put "imgL" here!!!
    except cv2.error:
        print('sosi')
        numDisparities = 144

    filteredImg = cv2.normalize(src=displ, dst=displ, beta=0, alpha=255, norm_type=cv2.NORM_MINMAX,dtype=cv2.CV_8U)
    toc = round(time.time(), 3)

    disp1 = filteredImg[302][331]
    print(disp1)
    # disp2 = filteredImg[350][500]
    # disp3 = filteredImg[150][400]

    depth = (0.004*0.072)/(0.263 * disp1) * 10000000
    # distance2 = round(((0.004 * 0.072/ disp2) * 100000) - 1, 3)
    # distance3 = round(((0.004 * 0.072/ disp3) * 100000) - 1, 3)

    cv2.circle(filteredImg, (250,500), 5, 255)
    # cv2.putText(filteredImg, str(distance1), (40, 320),fontFace=cv2.FAST_FEATURE_DETECTOR_FAST_N,fontScale= 1.0,color=(0,255,255))
    # cv2.putText(filteredImg, str(distance2), (350, 500),fontFace=cv2.FAST_FEATURE_DETECTOR_FAST_N,fontScale= 1.0,color=(0,255,255))
    # cv2.putText(filteredImg, str(distance3), (150, 400),fontFace=cv2.FAST_FEATURE_DETECTOR_FAST_N,fontScale= 1.0,color=(0,255,255))

    # print(toc - tic)
    print(depth)
    cv2.imshow('disparity', filteredImg)
    cv2.imshow('left', dst_left)
    cv2.imshow('right', dst_right)


    c = cv2.waitKey(5)
    if c == 27:
        break

cv2.waitKey()
cv2.destroyAllWindows()