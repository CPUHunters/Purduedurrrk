import numpy as np
import time
import cv2 as cv
import matplotlib.pyplot as plt
import os
import pretrained as ob
import time

START = 131
END = 231

def label(cap):
    while(1):
        ret,frame = cap.read()
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        ret, thresh = cv.threshold(gray,0,255,cv.THRESH_BINARY_INV+cv.THRESH_OTSU)
        
        kernel = np.ones((3,3),np.uint8)
        opening = cv.morphologyEx(thresh, cv.MORPH_OPEN,kernel, iterations =2)
        
        sure_bg = cv.dilate(opening, kernel, iterations=3)
        dist_transform=cv.distanceTransform(opening,cv.DIST_L2,5)
        ret, sure_fg = cv.threshold(dist_transform,0.7*dist_transform.max(),255,0)
        sure_fg = np.uint8(sure_fg)
        unknown = cv.subtract(sure_bg, sure_fg)
        ret, markers = cv.connectedComponents(sure_fg)
        markers = markers+1
        markers[unknown==255] = 0
        markers = cv.watershed(frame,markers)
        frame[markers == -1] = [255,0,0]
        frame=cv.resize(frame, (640,480), interpolation=cv.INTER_LINEAR)
        
        cv.imshow('frame',frame)
        k = cv.waitKey(30) & 0xff
        if k == 27:
            break
        
    cap.release()
    cv.destroyAllWindows()

def split(frame):
    upper_half = np.hsplit(np.vsplit(frame, 2)[0], 2)
    lower_half = np.hsplit(np.vsplit(frame, 2)[1], 2)
    
    upper_left = upper_half[0]
    upper_right = upper_half[1]
    lower_left = lower_half[0]
    lower_right = lower_half[1]

    return [upper_left, upper_right, lower_left, lower_right]

def bgSubMOG2(cap):
    fgbg = cv.createBackgroundSubtractorMOG2(history=500,varThreshold=500,detectShadows=0)
    
    str = 0
    count = 1
    while(count < (END+1)):
        if count==START:
            start_time=time.time()

        ret, frame = cap.read()
        _, original = cap.read()
        _, test = cap.read()

        if ret is True:
            hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
        else:
            continue
            
        lower_red=np.array([90,0,200])
        upper_red=np.array([255,255,255])
        #mask = cv.inRange(hsv, lower_red, upper_red)
        fmask = cv.inRange(hsv, lower_red, upper_red)
        fmask = cv.medianBlur(fmask, 3)
        
        #res = cv.bitwise_and(frame,frame,mask=mask)
        fres = cv.bitwise_and(frame, frame, mask=fmask) 

        #frame = cv.resize(frame, (640, 480), interpolation=cv.INTER_LINEAR)
        original = cv.resize(original, (640, 480), interpolation=cv.INTER_LINEAR)
        test = cv.resize(test, (640,480), interpolation=cv.INTER_LINEAR)
        #mask = cv.resize(mask, (640, 480), interpolation=cv.INTER_LINEAR)
        #fmask = cv.resize(fmask, (640, 480), interpolation=cv.INTER_LINEAR)
        #res = cv.resize(res, (640, 480), interpolation=cv.INTER_LINEAR)
        fres = cv.resize(fres, (640, 480), interpolation=cv.INTER_LINEAR)

        #fgmaskres = fgbg.apply(res)
        #fgmaskfra = fgbg.apply(frame)
        fgmaskfres = fgbg.apply(fres)
        #fgmaskfres = cv.medianBlur(fgmaskres, 3)   

        nlabels, labels, stats, centroids = cv.connectedComponentsWithStats(fgmaskfres)
        
        for index, centroid in enumerate(centroids):
            if stats[index][0] == 0 and stats[index][1] == 0:
                continue
            if np.any(np.isnan(centroid)):
                continue
            
            x, y, width, height, area = stats[index]
            centerX, centerY = int(centroid[0]), int(centroid[1])
            
            if area > 6:
                if count>= START and count <END:
                    if (x-20)>0 and (y-20)>0:
                        imgGrop = test[y-20:y+height+20,x-20:x+width+20]
                        path='C:/Purduedurrrk/PretrainedModel/object_detection/cropped'
                        cv.imwrite(os.path.join(path, '{}.jpg'.format(str)), imgGrop)
                        str = str + 1
                    
                #cv.circle(original, (centerX, centerY), 1, (0,255,0),2)
                #cv.rectangle(original, (x,y), (x + width, y + height),(0,0,255))
                #img_trim=img[y:y+height,x:x+width]
                #cv.imwrite("image.jpg",img_trim)

        #cv.imshow('frame_res',fgmaskres) # Green deleted
        #cv.imshow('original', original)
        #print(count)
        if count > END:
            print('END before', count)
            break
        
        print(count)
        count = count+1
        #cv.imshow('frame_original', fgmaskfra) # BS original
        #cv.imshow('res',res)
        #cv.imshow('frame_filter', fgmaskfres) # Median filtered green deleted
        #out.write(res)
        
        #if str != 0:
            #break

        k = cv.waitKey(30) & 0xff
        if k == 27:
            break
        
    ob.main()
    end_time = time.time()
    print("--- %.6s seconds ---"%(end_time-start_time))
    cap.release()
    #out.release()
    cv.destroyAllWindows()