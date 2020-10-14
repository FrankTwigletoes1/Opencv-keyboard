import cv2
import argparse
import numpy as np

def backSub():
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))


    backSub = cv2.createBackgroundSubtractorMOG2(detectShadows=False)
    #capture = cv.VideoCapture("randomimg.png") 
    capture = cv2.VideoCapture(1) 


    if not capture.isOpened:
        print('Unable to open: ' + args.input)
        exit(0)
    while True:
        ret, frame = capture.read()
        if frame is None:
            break
    
        fgMask = backSub.apply(frame, 0)
    
        fgMask = cv2.morphologyEx(fgMask, cv2.MORPH_OPEN, kernel)

        fgMask = cv2.medianBlur(fgMask, 15)

        contours, hierarchy = cv2.findContours(fgMask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

        for c in contours:
            if cv.contourArea(c) < 500:
                continue

            x,y,w,h = cv2.boundingRect(c)

            cv2.rectangle(frame, (x,y), (x + w, y + h), (255,0,0), 2)
   
        cv.imshow('Frame', frame)
        cv.imshow('FG Mask', fgMask)
    
        keyboard = cv2.waitKey(30)
        if keyboard == 'q' or keyboard == 27:
            break

def shi_tomashi(image):
    """
    Use Shi-Tomashi algorithm to detect corners
    Args:
        image: np.array
    Returns:
        corners: list
    """
    gray = cv2.cvtColor(cv2.medianBlur(image, 15), cv2.COLOR_RGB2GRAY)
    corners = cv2.goodFeaturesToTrack(gray, 4, 0.01, 100)
    corners = np.int0(corners)
    corners = sorted(np.concatenate(corners).tolist())
    print('\nThe corner points are...\n', corners)
    return corners
    

video_captured = cv2.VideoCapture(1)
#ret, frame = video_captured.read()
x = 100
while (True):
    # read frame-by-frame
    ret, frame = video_captured.read()
    corner1, corner2, corner3, corner4 = shi_tomashi(frame)
    
    cv2.circle(frame,tuple(corner1),5,(0,255,0), thickness = 3)
    cv2.circle(frame,tuple(corner2),5,(0,255,0), thickness = 3)
    cv2.circle(frame,tuple(corner3),5,(0,255,0), thickness = 3)
    cv2.circle(frame,tuple(corner4),5,(0,255,0), thickness = 3)

   
    
    
    cv2.imshow('Video', frame)

    
    x += 1
    if (cv2.waitKey(1) & 0xFF == ord('q')):
        break




