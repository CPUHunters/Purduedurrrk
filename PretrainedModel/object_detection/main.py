import cv2
import detection
import pretrained as pt

#cap = cv2.VideoCapture(0)
cap = cv2.VideoCapture('Test1.MOV')

while True:
    #detection.colorSpace(cap)
    
    #detection.bgSubMOG2(cap)
    #pt.main()

    pt.mog(cap)

    #detection.frameDiff(cap)
    #detection.opticalFlow(cap)
    #print("XX")
    #cnt = 0
    #detection.bgMeanFilter(cap, cnt)
    

