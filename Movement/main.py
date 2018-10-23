import cv2
import detection

#cap = cv2.VideoCapture(0)
cap = cv2.VideoCapture('NewTest1.mp4')

while True:
    #detection.colorSpace(cap)
    detection.bgSubMOG2(cap)
    #detection.frameDiff(cap)
    #detection.opticalFlow(cap)
    print("XX")

