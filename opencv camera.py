import cv2
import argparse
import numpy as np
from pysinewave import SineWave
import time


class Detection(object):
 
    THRESHOLD = 100
 
    def __init__(self, image):
        self.previous_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
 
    def get_active_cell(self, image):
        # obtain motion between previous and current image
        current_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        delta = cv2.absdiff(self.previous_gray, current_gray)
        threshold_image = cv2.threshold(delta, 25, 255, cv2.THRESH_BINARY)[1]
        
        cv2.imshow("Standard frame", image)
        cv2.imshow("Gray-scaled", current_gray)
        #cv2.imshow("Threshold", threshold_image)
        
        cv2.waitKey(1)
 
        # store current image
        self.previous_gray = current_gray
 
        # set cell width
        height, width = threshold_image.shape[:2]
        cell_width = int(width/7)
 
        # store motion level for each cell
        cells = np.array([0, 0, 0, 0, 0, 0, 0])
        cells[0] = cv2.countNonZero(threshold_image[0:200, 50:125])
        cells[1] = cv2.countNonZero(threshold_image[0:200, 125:200])
        cells[2] = cv2.countNonZero(threshold_image[0:200, 200:275])
        cells[3] = cv2.countNonZero(threshold_image[0:200, 275:350])
        cells[4] = cv2.countNonZero(threshold_image[0:200, 350:425])
        cells[5] = cv2.countNonZero(threshold_image[0:200, 425:500])
        cells[6] = cv2.countNonZero(threshold_image[0:200, 500:575])
        print(cells[0], " | ", cells[1], " | ", cells[2], " | ", cells[3], " | ", cells[4], " | ", cells[4], " | ", cells[5], " | ", cells[6], " | ")
        top_cell =  np.argmax(cells)


        # return the most active cell, if threshold met
        if(cells[top_cell] >= self.THRESHOLD):
            #print("play: ",top_cell)
            return top_cell
        else:
            return None

class main(object):
    def __init__(self):
        self.capture = cv2.VideoCapture(0)
        ret, frame = self.capture.read()
        self.detect = Detection(frame)
        self.sinewave = SineWave(pitch = int(0), pitch_per_second = int(1000), decibels=int(1))
    

    def draw_rectangles(self, frame):
        for x in range(0,7):
            cv2.rectangle(frame, (50+(x*75), 0), (50+(x*75+75),200), (255,255,255), 5)
        return frame


    def start(self):
        while True:
            self.ret, self.frame = self.capture.read()
            self.frame = self.draw_rectangles(self.frame)
            self.frame = cv2.flip(self.frame, 1)

            keyCheck = self.detect.get_active_cell(self.frame)
            

            if keyCheck == 0: self.sinewave.set_frequency(261.63)
            elif keyCheck == 1: self.sinewave.set_frequency(293.66)
            elif keyCheck == 2: self.sinewave.set_frequency(329.63)
            elif keyCheck == 3: self.sinewave.set_frequency(349.23)
            elif keyCheck == 4: self.sinewave.set_frequency(392)
            elif keyCheck == 5: self.sinewave.set_frequency(440.9)
            elif keyCheck == 6: self.sinewave.set_frequency(493.88)
            if keyCheck or keyCheck == 0: self.sinewave.play()
            

main = main()

main.start()





"""
backSub = cv2.createBackgroundSubtractorMOG2(detectShadows=False)

capture = cv2.VideoCapture(0) 

while True:
    ret, frame = capture.read()
    if frame is None:
        break

    frame = cv2.flip(frame, 1)
    
    fgMask = backSub.apply(frame, 0)

    fgMask = cv2.medianBlur(fgMask, 5)

    contours, hierarchy = cv2.findContours(fgMask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for x in range(0,7):
        cv2.rectangle(frame, (50+(x*75), 0), (50+(x*75+75),200), (0,255,0), 4)
    

    for c in contours:
        if cv2.contourArea(c) < 500:
            continue

        x,y,w,h = cv2.boundingRect(c)

        cv2.rectangle(frame, (x,y), (x + w, y + h), (255,0,0), 2)
   
    cv2.imshow('Frame', frame)
    
    keyboard = cv2.waitKey(1)
    if keyboard == 'q' or keyboard == 27:
        break



















#def shi_tomashi(image):
#    """
#    Use Shi-Tomashi algorithm to detect corners
#    Args:
#        image: np.array
#    Returns:
#        corners: list
#    """
#    #cv2.medianBlur(image, 15)
#    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
#    corners = cv2.goodFeaturesToTrack(gray, 4, 0.01, 100)
#    corners = np.int0(corners)
#    corners = sorted(np.concatenate(corners).tolist())
#    print('\nThe corner points are...\n', corners)
#    return corners
##
#video_captured = cv2.VideoCapture(0)
#ret, frame = video_captured.read()
#x = 100
#while (True):
#    # read frame-by-frame
#    ret, frame = video_captured.read()
#    try:
#        corner1, corner2, corner3, corner4 = shi_tomashi(frame)
#    except:
#        continue
#    cv2.circle(frame,tuple(corner1),5,(0,255,0), thickness = 3)
#    cv2.circle(frame,tuple(corner2),5,(0,255,0), thickness = 3)
#    cv2.circle(frame,tuple(corner3),5,(0,255,0), thickness = 3)
#    cv2.circle(frame,tuple(corner4),5,(0,255,0), thickness = 3)

   
    
    
#    cv2.imshow('Video', frame)

    
#    x += 1
#    if (cv2.waitKey(1) & 0xFF == ord('q')):
#        break




