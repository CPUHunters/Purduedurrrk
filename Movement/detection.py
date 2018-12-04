import numpy as np
import time
import cv2 as cv
import matplotlib.pyplot as plt
import os

def bgSubMOG2(cap):
    fgbg = cv.createBackgroundSubtractorMOG2(history=500, varThreshold=500, detectShadows=0)

    count = 1
    while (1):
        ret, frame = cap.read()
        _, original = cap.read()
        _, test = cap.read()
        if ret is True:
            hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
        else:
            continue

        lower_red = np.array([150, 0, 150])
        upper_red = np.array([255, 255, 255])
        mask = cv.inRange(hsv, lower_red, upper_red)
        fmask = cv.inRange(hsv, lower_red, upper_red)
        fmask = cv.medianBlur(fmask, 3)

        res = cv.bitwise_and(frame, frame, mask=mask)
        fres = cv.bitwise_and(frame, frame, mask=fmask)

        frame = cv.resize(frame, (640, 480), interpolation=cv.INTER_LINEAR)
        original = cv.resize(original, (640, 480), interpolation=cv.INTER_LINEAR)
        test = cv.resize(test, (640, 480), interpolation=cv.INTER_LINEAR)
        mask = cv.resize(mask, (640, 480), interpolation=cv.INTER_LINEAR)
        fmask = cv.resize(fmask, (640, 480), interpolation=cv.INTER_LINEAR)
        res = cv.resize(res, (640, 480), interpolation=cv.INTER_LINEAR)
        fres = cv.resize(fres, (640, 480), interpolation=cv.INTER_LINEAR)

        fgmaskres = fgbg.apply(res)
        fgmaskfra = fgbg.apply(frame)
        fgmaskfres = fgbg.apply(fres)
        fgmaskfres = cv.medianBlur(fgmaskres, 3)

        nlabels, labels, stats, centroids = cv.connectedComponentsWithStats(fgmaskres)
        str = 0
        for index, centroid in enumerate(centroids):
            if stats[index][0] == 0 and stats[index][1] == 0:
                continue
            if np.any(np.isnan(centroid)):
                continue

            x, y, width, height, area = stats[index]
            centerX, centerY = int(centroid[0]), int(centroid[1])

            if area > 40:
                if count == 15:
                    if (x-20) > 0 and (y-20) > 0:
                        imgGrop = test[y - 20:y + height + 20, x - 20:x + width + 20]
                        cv.imwrite("./test/%d.jpg" % str, imgGrop)
                        str = str + 1

                cv.circle(original, (centerX, centerY), 1, (0, 255, 0), 2)
                cv.rectangle(original, (x, y), (x + width, y + height), (0, 0, 255))

        cv.imshow('frame_res', fgmaskres)  # Green deleted
        cv.imshow('original', original)
        print(count)
        count = count + 1
        #cv.imshow('frame_original', fgmaskfra) # BS original
        cv.imshow('res',res)
        #cv.imshow('frame_filter', fgmaskfres) # Median filtered green deleted

        k = cv.waitKey(30) & 0xff
        if k == 27:
            break

    cap.release()
    # out.release()
    cv.destroyAllWindows()
