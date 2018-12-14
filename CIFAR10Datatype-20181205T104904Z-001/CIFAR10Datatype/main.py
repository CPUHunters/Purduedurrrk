import cv2
import detection

#cap = cv2.VideoCapture(0)
cap = cv2.VideoCapture('./data/Test1.MOV')

detection.bgSubMOG2(cap)


