'''
http://pythonopencv.com/driver-drowsiness-detection-using-opencv-and-python/
'''
from scipy.spatial import distance as dist
from imutils.video import VideoStream
from imutils import face_utils
from threading import Thread
#import numpy as np
import pyglet
import argparse
import imutils
import time
import dlib
import cv2

def sound_alarm(path):
    # play an alarm sound
    music = pyglet.resource.media('Alert_3.wav')
    #播放声音
    music.play()
    pyglet.app.run()

def eye_aspect_ratio(eye):
     # 计算上下眼皮的欧氏距离
	A = dist.euclidean(eye[1], eye[5])
	B = dist.euclidean(eye[2], eye[4])
 
     # 计算左右眼角点的欧式距离
	C = dist.euclidean(eye[0], eye[3])

	# 计算眼睛的纵横比
	ear = (A + B) / (2.0 * C)

	# return the eye aspect ratio
	return ear

# construct the argument parse and parse the arguments
# argparse模块用于解析命令行参数，详见https://www.cnblogs.com/zknublx/p/6106343.html
# 创建一个解析对象
ap = argparse.ArgumentParser()
# 每一个add_argument对应一个要关注的参数或选项：-w.--webcam是命令行参数名；type是参数类型；default是默认值；help是运行程序参数不正确时打印描述信息
#ap.add_argument("-p", "--shape-file", required=True,
	#help="path to facial landmark predictor")
#ap.add_argument("-a", "--alarm", type=str, default="",
	#help="path alarm.WAV file")
ap.add_argument("-w", "--webcam", type=int, default=0,
	help="index of webcam on system")
# 调用parse_args()方法进行解析，解析成功后可使用
args = vars(ap.parse_args())
 
# 初始化眼睛纵横比阈值
EYE_AR_THRESH = 0.23
# 初始化嘴巴开合的阈值
LIP_AR_THRESH = 25
# 初始化连续小于纵横比阈值的帧数值
EYE_AR_CONSEC_FRAMES = 15
# 初始化大于嘴巴开合阈值帧数
LIP_AR_CONSEC_FRAMES = 20

# 初始化眼睛帧数计数器
EYE_COUNTER = 0
# 初始化嘴巴帧数计数器
LIP_COUNTER = 0
ALARM_ON = False

# initialize dlib's face detector (HOG-based) and then create the facial landmark predictor
print("[INFO] loading facial landmark predictor...")
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# 左眼和右眼提取（x，y）坐标的起始和结束数组切片索引值
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

# 基于文件的视频流或实时USB/网络摄像头/ Raspberry Pi摄像头视频流
print("[INFO] starting video stream thread...")
#vs=cv2.VideoCapture('7-MaleGlasses.avi')
#vs = VideoStream(src=args["webcam"]).start()
vs = VideoStream(src=0).start()  #内置摄像头或USB摄像头
time.sleep(1.0)

# loop over frames from the video stream
while True:
    #从视频流循环帧
    frame = vs.read()
    #_,frame = vs.read()
    #frame = imutils.resize(frame, width=450)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # 检测人脸数
    rects = detector(gray, 0)
    
    # 在这些人脸中循环
    for rect in rects:
        # shape确定面部区域的面部标志，接着将这些（x，y）坐标转换成NumPy阵列
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)
        #print(shape)

        # 提取左右眼的坐标，用这些坐标计算眼睛的纵横比
        leftEye = shape[lStart:lEnd]
        #print(leftEye)
        rightEye = shape[rStart:rEnd]
        bottomlip=shape[48:59]
        leftEAR = eye_aspect_ratio(leftEye)
        rightEAR = eye_aspect_ratio(rightEye)
        lipDis = bottomlip[9] - bottomlip[3]
        

        # average the eye aspect ratio together for both eyes
        ear = (leftEAR + rightEAR) / 2.0

        # cv2.convexHull()函数检查一个曲线的凸性缺陷并进行修正
        leftEyeHull = cv2.convexHull(leftEye)
        rightEyeHull = cv2.convexHull(rightEye)
        bottomlipHull = cv2.convexHull(bottomlip)
        #绘制轮廓，使用cv2.drawContours函数
        #第一个参数是原图像，第二个参数是应该作为Python列表传递的轮廓，第三个参数是轮廓的索引（在绘制单个轮廓时有用）绘制所有轮廓，传递-1），剩余的参数是颜色，厚度 
        cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
        cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)
        cv2.drawContours(frame, [bottomlipHull], -1, (0, 255, 0), 1)


        # check to see if the eye aspect ratio is below the blink
        # threshold, and if so, increment the blink frame counter
        if ear < EYE_AR_THRESH:
            EYE_COUNTER += 1

            # if the eyes were closed for a sufficient number of
            # then sound the alarm
            if EYE_COUNTER >= EYE_AR_CONSEC_FRAMES:
                # if the alarm is not on, turn it on
                if not ALARM_ON:
                    ALARM_ON = True

                    # check to see if an alarm file was supplied,
                    # and if so, start a thread to have the alarm
                    # sound played in the background
                    if args["webcam"] != "":
                        t = Thread(target=sound_alarm,
                                   args=(args["webcam"],))
                        t.deamon = True
                        t.start()
                # draw an alarm on the frame
                # 各参数依次是：照片/添加的文字/左上角坐标/字体/字体大小/颜色/字体粗细
                cv2.putText(frame, "ALERT!", (200, 250),
                            cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 255), 5)

        # otherwise, the eye aspect ratio is not below the blink
        # threshold, so reset the counter and alarm
        else:
            EYE_COUNTER = 0
            ALARM_ON = False
        
        
        
        if lipDis[1] > LIP_AR_THRESH:
            LIP_COUNTER += 1

            # if the eyes were closed for a sufficient number of
            # then sound the alarm
            if LIP_COUNTER >= LIP_AR_CONSEC_FRAMES:
                # if the alarm is not on, turn it on
                if not ALARM_ON:
                    ALARM_ON = True

                    # check to see if an alarm file was supplied,
                    # and if so, start a thread to have the alarm
                    # sound played in the background
                    if args["webcam"] != "":
                        t = Thread(target=sound_alarm,
                                   args=(args["webcam"],))
                        t.deamon = True
                        t.start()
                # draw an alarm on the frame
                # 各参数依次是：照片/添加的文字/左上角坐标/字体/字体大小/颜色/字体粗细
                cv2.putText(frame, "ALERT!", (200, 250),
                            cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 255), 5)

        # otherwise, the eye aspect ratio is not below the blink
        # threshold, so reset the counter and alarm
        else:
            LIP_COUNTER = 0
            ALARM_ON = False


        # draw the computed eye aspect ratio on the frame to help
        # with debugging and setting the correct eye aspect ratio
        # thresholds and frame counters
        cv2.putText(frame, "EYE: {:.2f}".format(ear), (500, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(frame, "LIP: {:.2f}".format(lipDis[1]), (500, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    # show the frame
    cv2.imshow("video", frame)
    key = cv2.waitKey(1) & 0xFF
    
    # if the `q` key was pressed, break from the loop
    if key == ord("q"):
        break

# do a bit of cleanup
cv2.destroyAllWindows()
vs.stop()
