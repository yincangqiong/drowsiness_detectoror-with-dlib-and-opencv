# -*- coding: utf-8 -*-
"""
Created on Tus Jul 2 2018

@author: cangnqiong
"""
import cv2
import os
from PIL import Image

#视频转换成图片

cap=cv2.VideoCapture("10-FemaleNoGlasses.avi")  #读入视频
c=0
if cap.isOpened():
    ret,frame = cap.read() #分解为一帧一帧图像
else:
    ret=False
while ret:  #要对视频是否处理完成做判断
    ret,frame = cap.read()    
    c=c+1
    cv2.imwrite('image/'+str(c) + '.jpg',frame) #导出图片并存储为图像
    if cv2.waitKey(1)>=0 : 
        break         #显示图像时，延迟1ms再显示下一帧图片
cap.release()   #释放视频

'''
转成灰度图片

fin = 'image'  #要操作的文件夹
fout='image-Grayscale'
for file in os.listdir(fin):   #遍历文件夹中的图片
    file_fullname = fin + '/' +file
    img = Image.open(file_fullname).convert('L')
    #roi=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)   #转换为灰度图像
    #cv2.imshow('frame',gray)  #显示标记后的图像
    out_path = fout + '/' + file
    img.save(out_path)
'''

'''
裁剪图片大小

fin = 'image'  #要操作的文件夹
fout = 'image-resize' #resize后的存放文件夹
for file in os.listdir(fin):   #遍历文件夹中的图片
    file_fullname = fin + '/' +file
    img = Image.open(file_fullname)
    roi=img.resize((380,380))   #将图片裁剪成250*250
    out_path = fout + '/' + file
    roi.save(out_path)
'''
          
