# -*- coding: utf-8 -*-
"""
Created on Wed Jul  4 15:15:27 2018

@author: gz04179
"""

import dlib
from skimage import io
import cv2

# 使用Dlib的正面人脸检测器frontal_face_detector
detector = dlib.get_frontal_face_detector()

# dlib的68点模型
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
 
# cv2读取图像
img = cv2.imread("1.jpg")

# 生成dlib的图像窗口
win = dlib.image_window()
win.set_image(img)

# 使用detector检测器来检测图像中的人脸
dets = detector(img, 1)
print("人脸数：", len(dets))

for i, d in enumerate(dets):
    print("第", i+1, "个人脸的矩形框坐标：",
          "left:", d.left(), "right:", d.right(), "top:", d.top(), "bottom:", d.bottom())

    # 使用predictor来计算面部轮廓
    shape = predictor(img, dets[i])
    # 绘制面部轮廓
    win.add_overlay(shape)

# 绘制矩阵轮廓
win.add_overlay(dets)