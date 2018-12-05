import numpy as np
import time
import cv2 as cv
import matplotlib.pyplot as plt
import os
import retrain_run
import shutil
import time
START=175
END=181

def bgSubMOG2(cap):
    fgbg = cv.createBackgroundSubtractorMOG2(history=500, varThreshold=500, detectShadows=0)

    count = 1
    while (count<(END+1)):
        if count==START:
            start_time=time.time()

        ret, frame = cap.read()
        _, original = cap.read()
        _, test = cap.read()

        if ret is True:
            hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
        else:
            continue

        lower_red = np.array([90, 0, 200])
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
        if count >=START and count<END:
            if not (os.path.isdir("./test/%s" % count)):
                os.makedirs(os.path.join("./test/%s" % count))
        for index, centroid in enumerate(centroids):
            if stats[index][0] == 0 and stats[index][1] == 0:
                continue
            if np.any(np.isnan(centroid)):
                continue
            x, y, width, height, area = stats[index]
            centerX, centerY = int(centroid[0]), int(centroid[1])
            if area > 6:
                if count>= START and count <END:
                    if (x-20) > 0 and (y-20) > 0:
                        imgGrop = test[y - 20:y + height + 20, x - 20:x + width + 20]
                        cv.imwrite("./test/%s/%d.jpg" % (count,str), imgGrop)
                        #cv.imwrite("./test1/%d-%d.jpg" % (count, str), imgGrop)
                        str = str + 1

                cv.circle(original, (centerX, centerY), 1, (0, 255, 0), 2)
                cv.rectangle(original, (x, y), (x + width, y + height), (0, 0, 255))
        if count >=START and count<END:
            retrain_run.run_inference_on_image(count)
        if count==END:
            end_time=time.time()
        #shutil.rmtree('./test')
        #cv.imshow('frame_res', fgmaskres)  # Green deleted
        #cv.imshow('original', original)
        #print(count)
        count = count + 1
        #cv.imshow('frame_original', fgmaskfra) # BS original
        #cv.imshow('res',res)
        #cv.imshow('frame_filter', fgmaskfres) # Median filtered green deleted
        k = cv.waitKey(30) & 0xff
        if k == 27:
            break
    print("--- %.6s seconds ---"%(end_time-start_time))

    cap.release()
    # out.release()
    cv.destroyAllWindows()
