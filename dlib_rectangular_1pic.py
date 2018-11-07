# -*- coding: utf-8 -*-
"""
Created on Wed Jul  4 11:32:36 2018

@author: cangqiong
"""

import dlib
import cv2
from skimage import io

# 使用Dlib的正面人脸检测器frontal_face_detector
detector = dlib.get_frontal_face_detector()

# cv2读取图像
img= cv2.imread("1.jpg")
 
# 生成dlib的图像窗口
win = dlib.image_window()
win.set_image(img)


# 使用detector检测器来检测图像中的人脸,len(dets)代表人脸数
dets = detector(img, 1)

for i,d in enumerate(dets):
    print("人脸的矩形框坐标：",
          "left:", d.left(), "right:", d.right(), "top:", d.top(), "bottom:", d.bottom())
 
# 绘制矩阵轮廓
win.add_overlay(dets)