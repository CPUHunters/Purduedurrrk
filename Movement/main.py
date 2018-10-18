import cv2
import detection

cap = cv2.VideoCapture(0)
#cap = cv2.VideoCapture('IMG_3867.MOV')

while True:
    #detection.colorSpace(cap)
    #detection.bgSub(cap)
    detection.frameDiff(cap)
    #detection.opticalFlow(cap)
    print("XX")

