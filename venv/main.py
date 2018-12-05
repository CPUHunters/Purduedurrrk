import cv2
import detection
import retrain_run

#cap = cv2.VideoCapture(0)
cap = cv2.VideoCapture('Test1.MOV')

    #detection.colorSpacee(cap)
detection.bgSubMOG2(cap)
    #detection.label(cap)
    #detection.frameDiff(cap)
    #detection.opticalFlow(cap)
    #print("XX")

