# -*- coding: utf-8 -*-
"""
Created on Fri Jul 27 17:35:15 2018

@author: gz04179
"""

import cv2
import dlib

cap=cv2.VideoCapture("10-FemaleNoGlasses.avi")  #读入视频

#源程序是用sys.argv从命令行参数去获取训练模型，精简版我直接把路径写在程序中了
predictor_path = "shape_predictor_68_face_landmarks.dat"

#使用官方提供的模型构建特征提取器
predictor = dlib.shape_predictor(predictor_path)

#使用dlib自带的frontal_face_detector作为人脸检测器
detector = dlib.get_frontal_face_detector()

while True:
    _,frame=cap.read()
    dets = detector(frame, 1)
    if len(dets) != 0:
        shape = predictor(frame, dets[0])
        for p in shape.parts():
            #cv2.circle()参数：图片，圆点，半径，颜色，空心1/实心-1
            cv2.circle(frame, (p.x, p.y), 1, (0,0,0), 1)
            
    cv2.imshow('video',frame)
    if cv2.waitKey(1)&0xFF==27:
        break
cap.release()
cv2.destroyAllWindows()