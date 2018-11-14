import numpy as np
import time
import cv2 as cv
import matplotlib.pyplot as plt

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


def colorSpace(cam):
    time.sleep(1)

    print("DETECTION IN PROCESS")

    while True:
        _, frame = cam.read()

        hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)

        #find the color space of bird. It should work
        #currently black
        #in the case for deer?

        lower_black = np.array([0, 0, 0])
        upper_black = np.array([180, 255, 30])

        mask = cv.inRange(hsv, lower_black, upper_black)

        if cv.sumElems(mask)[0] > 20: # the sum of the black pixels.
            print('detected')

        cv.imshow("Frame", frame)
        cv.imshow('Mask', mask)

        key = cv.waitKey(1) & 0xFF

        if key == ord("q"):
            break

def bgSubMOG(cap):
	fgbg = cv.bgsegm.createBackgroundSubtractorMOG()

	while(1):
		ret, frame=cap.read()
		_, original=cap.read()
		
		hsv=cv.cvtColor(frame, cv.COLOR_BGR2HSV)
                lower_red=np.array([90,0,200])
                upper_red=np.array([255,255,255])

                mask=cv.inRange(hsv, lower_red, upper_red)
                res=cv.bitwise_and(frame,frame,mask=mask)

                frame = cv.resize(frame, (640, 480), interpolation=cv.INTER_LINEAR)
		original = cv.resize(original, (640, 480), interpolation=cv.INTER_LINEAR)
		mask = cv.resize(mask, (640,480), interpolation=cv.INTER_LINEAR)
                res = cv.resize(res, (640,480), interpolation=cv.INTER_LINEAR)
                
                fgmask = fgbg.apply(frame)
		fgmask_res = fgbg.apply(res)

                cv.imshow('frame',fgmask)
		cv.imshow('original', original)
                cv.imshow('frame_res', fgmask_res)
                cv.imshow('res', res)
		k = cv.waitKey(30) & 0xff
		if k == 27:
			break
	cap.release()
	cv.destroyAllWindows()


def bgSubMOG2(cap):
	fgbg = cv.createBackgroundSubtractorMOG2(history=500,varThreshold=500,detectShadows=0)
	#fgbg.setDetectShadows(False)
        #fcc = cv.VideoWriter_fourcc(*'DIVX')
        #out = cv.VideoWriter('testout.avi',fcc,20.0,(640,480))

	while(1):
		ret, frame = cap.read()
	    	_, original = cap.read()

	        if ret is True:
                    hsv=cv.cvtColor(frame, cv.COLOR_BGR2HSV)
     		else:
                    continue
                #lower_red=np.array([0,20,20])
	        #upper_red=np.array([90,255,255])

	        lower_red=np.array([90,0,200])
        	upper_red=np.array([255,255,255])
                mask = cv.inRange(hsv, lower_red, upper_red)
		fmask = cv.inRange(hsv, lower_red, upper_red)
		fmask = cv.medianBlur(fmask, 3)
                
      		res = cv.bitwise_and(frame,frame,mask=mask)
     		fres = cv.bitwise_and(frame, frame, mask=fmask) 
		
		frame = cv.resize(frame, (640, 480), interpolation=cv.INTER_LINEAR)
		original = cv.resize(original, (640, 480), interpolation=cv.INTER_LINEAR)
  		mask = cv.resize(mask, (640, 480), interpolation=cv.INTER_LINEAR)
        	fmask = cv.resize(fmask, (640, 480), interpolation=cv.INTER_LINEAR)
		res = cv.resize(res, (640, 480), interpolation=cv.INTER_LINEAR)
		fres = cv.resize(fres, (640, 480), interpolation=cv.INTER_LINEAR)
                
      		fgmaskres = fgbg.apply(res)
      		fgmaskfra = fgbg.apply(frame)
        	fgmaskfres = fgbg.apply(fres)
        	fgmaskfres = cv.medianBlur(fgmaskres, 3)

                '''
                ret, markers = cv.connectedComponents(fgmaskres)
                markers = markers+1
                markers[unknown==255] = 0
                markers = cv.watershed(frame,markers)
                frame[markers == -1] = [255,0,0]
                '''
                nlabels, labels, stats, centroids = cv.connectedComponentsWithStats(fgmaskres)

                for index, centroid in enumerate(centroids):
                    if stats[index][0] == 0 and stats[index][1] == 0:
                        continue
                    if np.any(np.isnan(centroid)):
                        continue
                    
                    x, y, width, height, area = stats[index]
                    centerX, centerY = int(centroid[0]), int(centroid[1])

                    if area > 1:
                        cv.circle(original, (centerX, centerY), 1, (0,255,0),2)
                        cv.rectangle(original, (x,y), (x + width, y + height),(0,0,255))



        	cv.imshow('frame_res',fgmaskres) # Green deleted
		cv.imshow('original', original)
        	#cv.imshow('frame_original', fgmaskfra) # BS original
        	#cv.imshow('res',res)
        	#cv.imshow('frame_filter', fgmaskfres) # Median filtered green deleted
	        
                #out.write(res)
		
                k = cv.waitKey(30) & 0xff
		if k == 27:
        		break

	cap.release()
        #out.release()
	cv.destroyAllWindows()


def bgSubGMG(cap):

    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (3,3))
    fgbg = cv.bgsegm.createBackgroundSubtractorGMG()

    while(1):
        ret, frame = cap.read()
	_, original = cap.read()

        hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
        lower_red = np.array([90,0,200])
        upper_red = np.array([255,255,255])

        mask = cv.inRange(hsv, lower_red, upper_red)
        res = cv.bitwise_and(frame, frame, mask=mask)

	frame = cv.resize(frame, (640, 480), interpolation=cv.INTER_LINEAR)
	original = cv.resize(frame, ( 640, 480), interpolation=cv.INTER_LINEAR)
        mask = cv.resize(mask, (640, 480), interpolation=cv.INTER_LINEAR)
	res = cv.resize(res, (640, 480), interpolation=cv.INTER_LINEAR)
                
    #    frame = split(frame)[0]

        fgmask = fgbg.apply(frame)
        fgmask = cv.morphologyEx(fgmask, cv.MORPH_OPEN, kernel)
        fgmask_res = fgbg.apply(res)
        fgmask_res = cv.morphologyEx(fgmask_res, cv.MORPH_OPEN, kernel)

        cv.imshow('frame', fgmask)
        cv.imshow('original', original)
        cv.imshow('frame_res', fgmask_res)

    #    if cv.sumElems(fgmask)[0] > 100000:
     #       print('!')
      #      return frame

        key = cv.waitKey(30) & 0xff
        if key == 27:
            break

    cap.release()
    cv.destroyAllWindows()


def frameDiff(camera):
    time.sleep(1)
    cflag = 1
    cnt = 0

    t = 100

    print("DETECTION IN PROCESS")

    while True:
        _, image1 = camera.read()
        _, image2 = camera.read()
	image1 = cv.resize(image1, (640, 480), interpolation=cv.INTER_LINEAR)
	image2 = cv.resize(image2, (640, 480), interpolation=cv.INTER_LINEAR)

#        image1 = split(image1)[0]
#        image2 = split(image2)[0]

        grscale1 = cv.cvtColor(image1, cv.COLOR_BGR2GRAY)
        grscale2 = cv.cvtColor(image2, cv.COLOR_BGR2GRAY)

        d = cv.absdiff(grscale1, grscale2)
        ret, thresh = cv.threshold(d, t, 255, cv.THRESH_BINARY)

        if cv.sumElems(thresh)[0] > 200:
            cnt += 1

        elif cflag == 1:
            cflag = 0

        else:
            cflag = 1
            cnt = 0

        if cnt > 2:
            print("**considerable motion**")
            return image2
        cv.imshow("Frame", image1)
        cv.imshow("Motion", thresh)
        key = cv.waitKey(1) & 0xFF
        if key == ord("q"):
            print("END OF DETECTION")
            break

        if key == ord("s"):
            t = input("input threshold val (currently %d)"%t)


def opticalFlow(cap):
    # params for ShiTomasi corner detection
    feature_params = dict( maxCorners = 100,
                           qualityLevel = 0.1,
                           minDistance = 7,
                           blockSize = 7 )

    # Parameters for lucas kanade optical flow
    lk_params = dict( winSize  = (15,15),
                      maxLevel = 2,
                      criteria = (cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.03))

    # Create some random colors
    color = np.random.randint(0,255,(100,3))

    # Take first frame and find corners in it
    ret, old_frame = cap.read()
    old_frame = split(old_frame)[0]
    old_gray = cv.cvtColor(old_frame, cv.COLOR_BGR2GRAY)
    p0 = cv.goodFeaturesToTrack(old_gray, mask = None, **feature_params)

    # Create a mask image for drawing purposes
    mask = np.zeros_like(old_frame)

    while(1):
        ret,frame = cap.read()
        frame = split(frame)[0]
        frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

        # calculate optical flow
        p1, st, err = cv.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)

        # Select good points
        good_new = p1[st==1]
        good_old = p0[st==1]

        # draw the tracks
        for i,(new,old) in enumerate(zip(good_new,good_old)):
            a,b = new.ravel()
            c,d = old.ravel()
            mask = cv.line(mask, (a,b),(c,d), color[i].tolist(), 2)
            frame = cv.circle(frame,(a,b),5,color[i].tolist(),-1)

        img = cv.add(frame,mask)
        cv.imshow('frame',img)
        k = cv.waitKey(30) & 0xff

        if k == 27:
            break
