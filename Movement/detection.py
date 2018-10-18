import numpy as np
import time
import cv2 as cv

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
		frame = cv.resize(frame, (640, 480), interpolation=cv.INTER_LINEAR)
		original = cv.resize(original, (640, 480), interpolation=cv.INTER_LINEAR)
		fgmask = fgbg.apply(frame)
		cv.imshow('frame',fgmask)
		cv.imshow('original', original)
		k = cv.waitKey(30) & 0xff
		if k == 27:
			break
	cap.release()
	cv.destroyAllWindows()


def bgSubMOG2(cap):
	fgbg = cv.createBackgroundSubtractorMOG2()

	while(1):
		ret, frame = cap.read()
		_, original = cap.read()

		frame = cv.resize(frame, (640, 480), interpolation=cv.INTER_LINEAR)
		original = cv.resize(original, (640, 480), interpolation=cv.INTER_LINEAR)

		fgmask = fgbg.apply(frame)

		cv.imshow('frame',fgmask)
		cv.imshow('original', original)

		k = cv.waitKey(30) & 0xff
		if k == 27:
			break

	cap.release()
	cv.destroyAllWindows()


def bgSubGMG(cap):

    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (3,3))
    fgbg = cv.bgsegm.createBackgroundSubtractorGMG()

    while(1):
        ret, frame = cap.read()
	_, original = cap.read()

	frame = cv.resize(frame, (640, 480), interpolation=cv.INTER_LINEAR)
	original = cv.resize(frame, ( 640, 480), interpolation=cv.INTER_LINEAR)
    #    frame = split(frame)[0]

        fgmask = fgbg.apply(frame)
        fgmask = cv.morphologyEx(fgmask, cv.MORPH_OPEN, kernel)

        cv.imshow('frame', fgmask)
        cv.imshow('original', original)

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

