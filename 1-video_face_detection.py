# coding=utf-8
import cv2
import dlib
import time
import numpy as np

#使用dlib的人脸检测器
detector = dlib.get_frontal_face_detector()  #使用默认的人类识别器模型
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

def discern(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #使用detector检测器来检测图像中的人脸,len(dets)代表人脸数    
    dets = detector(gray, 1)
    
    for face in dets:
        left = face.left()
        top = face.top()
        right = face.right()
        bottom = face.bottom()
        cv2.rectangle(img, (left, top), (right, bottom), (0, 255, 0), 2)
        
        
        #landmarks中存储的就是68个点的坐标     
        landmarks = np.matrix([[p.x, p.y] for p in predictor(img,dets[0]).parts()])
        for idx, point in enumerate(landmarks):
            # 68点的坐标
            pos = (point[0, 0], point[0, 1])
            #print(idx,pos)
            # 利用cv2.circle给每个特征点画一个圈，共68个
            cv2.circle(img, pos, 1, color=(0, 0, 0))
            #利用cv2.putText输出1-68
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(img, str(idx+1), pos, font, 0.1, (0, 0, 255), 1,cv2.LINE_AA)
        cv2.imshow("image", img)
        
        

            
cap = cv2.VideoCapture('10-FemaleNoGlasses.avi')
while (1):
    ret, img = cap.read()
    #start_time=time.time()
    discern(img)
    #end_time=time.time()
    #print(start_time-end_time)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
