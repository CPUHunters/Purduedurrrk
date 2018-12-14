import cv2
import detection
import retrain_run


#cap = cv2.VideoCapture(0)
cap = cv2.VideoCapture('Test1.MOV')
#detection.colorSpacee(cap)
retrain_run.create_graph()
detection.bgSubMOG2(cap)
#retrain_run.run_inference_on_image(1)    
#detection.label(cap)
#detection.frameDiff(cap)
#detection.opticalFlow(cap)
#print("XX")
