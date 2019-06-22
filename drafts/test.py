import sys
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
import cv2
import numpy as np
import os


class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.color = QLabel()
        self.gray = QLabel()
        path1 = 'D:\learn\8 sem\diplom\diplom\left.jpg'
        self.img = cv2.imread(path1)
        self.gray_image = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
        self.initui()

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

    def initui(self):
        hbox = QHBoxLayout()
        self.color.setPixmap(QPixmap.fromImage(self.get_frame(self.img,onechannel=False)))
        self.gray.setPixmap(QPixmap.fromImage(self.get_frame(self.gray_image,onechannel=True)))

        hbox.addWidget(self.color,Qt.AlignCenter)
        hbox.addWidget(self.gray,Qt.AlignCenter)

        self.setLayout(hbox)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    mw = MainWindow()
    mw.show()
    sys.exit(app.exec_())