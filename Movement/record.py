import cv2 as cv
import numpy as np
import time

cap = cv.VideoCapture(0)

fourcc = cv.VideoWriter_fourcc(*'XVID')
out = cv.VideoWriter('out.avi', fourcc, 20.0, (640, 480))

while cap.isOpened():
    ret, frame = cap.read()

    if ret:
        out.write(frame)
        cv.imshow('frame', frame)

        if cv.waitKey(1) & 0xff == ord('q'):
            break
    else:
        break

cap.release()
out.release()
cv.destroyAllWindows()

