import cv2
from stereovision.calibration import StereoCalibration
import time
import numpy as np
import os


cv2.namedWindow('disparity', cv2.WINDOW_AUTOSIZE)
# left_path = os.path.join('D:\learn\8 sem\diplom\diplom', 'left1.jpg')
# right_path = os.path.join('D:\learn\8 sem\diplom\diplom', 'right.jpg')

calibrat = StereoCalibration(input_folder='D:\learn\8 sem\diplom\diplom\calib_result')
left_cam = cv2.VideoCapture(1)
right_cam = cv2.VideoCapture(2)

maxDisp = 0
wsize = 9

w,h = 640, 480
while True:
    # f1, frame_L = left_cam.read()
    # f2, frame_R = right_cam.read()
    frame_L = cv2.imread(os.path.join('D:\learn\8 sem\diplom\diplom', 'resources\pictures', 'left1.jpg'))
    frame_R = cv2.imread(os.path.join('D:\learn\8 sem\diplom\diplom', 'resources\pictures', 'lright1.jpg'))
    rectified_pair = calibrat.rectify((frame_L, frame_R))
    left = cv2.cvtColor(rectified_pair[0],cv2.COLOR_BGR2GRAY)
    right = cv2.cvtColor(rectified_pair[1],cv2.COLOR_BGR2GRAY)

    # left = rectified_pair[0]
    # right = rectified_pair[1]

    window_size = 5
    lmbda = 80000
    sigma = 1.2
    visual_multiplier = 1.0

    tic = round(time.time(), 3)
    # print('create matchers')
    left_matcher = cv2.StereoSGBM_create(minDisparity=0,
                         numDisparities=192,  # max_disp has to be dividable by 16 f. E. HH 192, 256
                         blockSize=window_size,
                         P1=8 * 3 * window_size ** 2,
                         # wsize default 3; 5; 7 for SGBM reduced size image; 15 for SGBM full size image (1300px and above); 5 Works nicely
                         P2=32 * 3 * window_size ** 2,
                         disp12MaxDiff=1,
                         uniquenessRatio=10,
                         speckleWindowSize=100,
                         speckleRange=32,
                         mode=False)
    right_matcher = cv2.ximgproc.createRightMatcher(left_matcher)

    # print('create filter')
    wls_filter = cv2.ximgproc.createDisparityWLSFilter(matcher_left=left_matcher)
    wls_filter.setLambda(lmbda)
    wls_filter.setSigmaColor(sigma)

    # print('computing disparity...')
    displ = left_matcher.compute(left, right)  # .astype(np.float32)/16
    dispr = right_matcher.compute(left, right)  # .astype(np.float32)/16
    filteredImg = wls_filter.filter(displ, left, None, dispr)  # important to put "imgL" here!!!

    filteredImg = cv2.normalize(src=filteredImg, dst=filteredImg, beta=0, alpha=255, norm_type=cv2.NORM_MINMAX,dtype=cv2.CV_8U)
    toc = round(time.time(), 3)

    disp = filteredImg[w // 2, h//2]
    distance = round(0.4 * 7.2 / disp,3)
    cv2.putText(filteredImg,str(distance), (w // 2, h//2),cv2.FONT_HERSHEY_COMPLEX,1,(0,255,255))
    print(toc - tic)

    cv2.imshow('disparity',filteredImg)
    cv2.imshow('left',left)
    cv2.imshow('right',right)
    c = cv2.waitKey(5)
    if c == 27:
        break
left_cam.release()
right_cam.release()
cv2.destroyAllWindows()

# import cv2
# from stereovision.calibration import StereoCalibration
# import time
# import numpy as np
#
#
# cv2.namedWindow('disparity', cv2.WINDOW_AUTOSIZE)
# # left_path = os.path.join('D:\learn\8 sem\diplom\diplom', 'left1.jpg')
# # right_path = os.path.join('D:\learn\8 sem\diplom\diplom', 'right.jpg')
#
# calibrat = StereoCalibration(input_folder='D:\learn\8 sem\diplom\diplom\calib_result')
# left_cam = cv2.VideoCapture(1)
# right_cam = cv2.VideoCapture(2)
#
# maxDisp = 0
# wsize = 9
#
# w,h = 640, 480
# while True:
#     f1, frame_L = left_cam.read()
#     f2, frame_R = right_cam.read()
#
#     rectified_pair = calibrat.rectify((frame_L, frame_R))
#     left = cv2.cvtColor(rectified_pair[0],cv2.COLOR_BGR2GRAY)
#     right = cv2.cvtColor(rectified_pair[1],cv2.COLOR_BGR2GRAY)
#
#     # left = rectified_pair[0]
#     # right = rectified_pair[1]
#
#     window_size = 5
#     lmbda = 80000
#     sigma = 1.2
#     visual_multiplier = 1.0
#
#     tic = round(time.time(), 3)
#     # print('create matchers')
#     left_matcher = cv2.StereoSGBM_create(minDisparity=-64,
#                          numDisparities=192,  # max_disp has to be dividable by 16 f. E. HH 192, 256
#                          blockSize=window_size,
#                          P1=8 * 3 * window_size ** 2,
#                          # wsize default 3; 5; 7 for SGBM reduced size image; 15 for SGBM full size image (1300px and above); 5 Works nicely
#                          P2=32 * 3 * window_size ** 2,
#                          disp12MaxDiff=1,
#                          uniquenessRatio=10,
#                          speckleWindowSize=100,
#                          speckleRange=32,
#                          mode=False)
#     right_matcher = cv2.ximgproc.createRightMatcher(left_matcher)
#
#     # print('create filter')
#     wls_filter = cv2.ximgproc.createDisparityWLSFilter(matcher_left=left_matcher)
#     wls_filter.setLambda(lmbda)
#     wls_filter.setSigmaColor(sigma)
#
#     # print('computing disparity...')
#     displ = left_matcher.compute(left, right)  # .astype(np.float32)/16
#     dispr = right_matcher.compute(left, right)  # .astype(np.float32)/16
#     displ = np.int16(displ)
#     dispr = np.int16(dispr)
#     filteredImg = wls_filter.filter(displ, left, None, dispr)  # important to put "imgL" here!!!
#
#     filteredImg = cv2.normalize(src=filteredImg, dst=filteredImg, beta=0, alpha=255, norm_type=cv2.NORM_MINMAX)
#     filteredImg = np.uint8(filteredImg)
#     toc = round(time.time(), 3)
#
#     disp = filteredImg[w // 2, h//2]
#     distance = round(0.4 * 7.2 / disp,3)
#     cv2.putText(filteredImg,str(distance), (w // 2, h//2),cv2.FONT_HERSHEY_COMPLEX,1,(0,255,255))
#     print(toc - tic)
#
#     cv2.imshow('disparity',filteredImg)
#     cv2.imshow('left',left)
#     cv2.imshow('right',right)
#     c = cv2.waitKey(5)
#     if c == 27:
#         break
# left_cam.release()
# right_cam.release()
# cv2.destroyAllWindows()