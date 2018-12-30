import numpy as np
import time
import cv2 as cv
import matplotlib.pyplot as plt
import os
import retrain_run
import shutil
import time
import pygame

START=130
END=140



def bgSubMOG2(cap):
    fgbg = cv.createBackgroundSubtractorMOG2(history=500, varThreshold=500, detectShadows=0)
    pygame.init()
    pygame.mixer.music.load('Hawk.mp3')
    count = 1                       #count는 프레임 번호
    while (count<(END+20)):

        # if count==START:            #시작 순간부터 시간 측정
        #     start_time=time.time()

        ret, frame = cap.read()
        _, original = cap.read()
        _, test = cap.read()

        if ret is True:
            hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
        else:
            continue

        lower_red = np.array([90, 0, 200])          #나뭇잎 컬러의 하위영역 설정
        upper_red = np.array([255, 255, 255])       #나뭇잎 컬러의 상위영역 설정
        mask = cv.inRange(hsv, lower_red, upper_red)        #mask에 색상영역 추출
        fmask = cv.inRange(hsv, lower_red, upper_red)       #fmask에 색상영역 추출(미디안 필터를 적용하기 위해 따로 적용)
        fmask = cv.medianBlur(fmask, 3)                     #median 필터를 fmask에 적용

        res = cv.bitwise_and(frame, frame, mask=mask)       #나뭇잎 영역을 제외한 부분 추출
        fres = cv.bitwise_and(frame, frame, mask=fmask)     #미디안 필터를 적용한 나뭇잎 영역을 제외한 부분 추출

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
            if not (os.path.isdir("./test/%s" % count)):            #해당 프레임에 대한 폴더 생성
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
                        str = str + 1

                cv.circle(original, (centerX, centerY), 1, (0, 255, 0), 2)
                cv.rectangle(original, (x, y), (x + width, y + height), (0, 0, 255))
        if count>= START and count < END:
            sound=retrain_run.run_inference_on_image(count)
            if sound ==1 and (not pygame.mixer.music.get_busy()):
                pygame.mixer.music.play()


        # if count==END:                      #시간 측정 완료
        #     end_time=time.time()

        count = count + 1


        #cv.imshow('frame_res', fgmaskres)  # Green deleted
        cv.imshow('original', original)
        #cv.imshow('frame_original', fgmaskfra) # BS original
        #cv.imshow('res',res)
        cv.imshow('frame_filter', fgmaskfres) # Median filtered green deleted

        k = cv.waitKey(30) & 0xff
        if k == 27:
            break

    # print("--- %.6s seconds ---"%(end_time-start_time))     #소요 시간 출력

    cap.release()
    cv.destroyAllWindows()
