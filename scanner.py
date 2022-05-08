# This Python file uses the following encoding: utf-8
import os
import time
import io
import threading
import calendar
from collections import deque
import matplotlib.pyplot as plt

from windowcapture import WindowCapture
from PySide6 import QtGui
from PySide6 import QtCore

import pytesseract
import cv2
import numpy as np
from playsound import playsound
from lz.reversal import reverse
from win32com.shell import shell, shellcon
import logging
import datetime
import math

class Scanner:
    def __init__(self, main_window):
        self.main_window=main_window
        self.ui = main_window.ui
        self.ui_scale = self.ui.ui_scale_spinner.value()
        self.ui_rotation = -3.65/self.ui_scale
        self.scan_scale = 5/self.ui_scale
        pytesseract.pytesseract.tesseract_cmd = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'Tesseract-OCR\\tesseract.exe')
        self.text_color = main_window.text_color_hsv
        self.icon_color = main_window.icon_color_hsv
        self.max_time = 120 if self.ui.time_120_radio_button.isChecked() else 156
        self.template_match_threshold = self.ui.template_match_slider.value()*0.6/100
        self.template_matching_enabled = False if self.template_match_threshold == 0 else True

        self.figure, self.axis = None,None

        self.M = cv2.getRotationMatrix2D((0,0), self.ui_rotation, 1)

        self.smeeta_proc_100_processed = cv2.imread(os.path.join(os.path.dirname(os.path.abspath(__file__)),'Templates\\smeeta_proc_100_processed.png'), 0 )

        self.wincap = WindowCapture('Warframe', ( 750 , 300 ) , self.ui)

        if self.text_color == [0,0,255]:
            self.text_hsv_filter = self.white_hsv_filter
        else:
            self.text_hsv_filter = self.hsv_filter

        self.procs_active = 0
        self.proc_expiry_queue = deque([]) 
        self.proc_warn_queue = deque([])

        self.sound_queue = deque([])

        self.disply_width = 300
        self.display_height = 50

    # Hue range is [0,179], Saturation range is [0,255] and Value range is [0,255].
    def hsv_filter(self, image, color, h_sens=5, s_sens=40, v_scale=0.3):
        image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        HSV_low = np.array([self.limit_color(color[0]-h_sens,179), self.limit_color(color[1]-s_sens), self.limit_color(color[2]*v_scale)], dtype=np.uint8)
        HSV_high = np.array([self.limit_color(color[0]+h_sens), self.limit_color(color[1]+1), self.limit_color(color[2]+40)], dtype=np.uint8)
        inverted = cv2.bitwise_not(cv2.inRange(image_hsv, HSV_low, HSV_high))
        return inverted

    def white_hsv_filter(self, image, color, h_sens=5, s_sens=40, v_scale=0.5):
        image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        HSV_low = np.array([0, 0, 150], dtype=np.uint8)
        HSV_high = np.array([179, 26, 255], dtype=np.uint8)
        inverted = cv2.inRange(image_hsv, HSV_low, HSV_high)
        return cv2.bitwise_not(inverted)

    def limit_color(self, value, max_v=255):
        if value <0:
            return 0
        elif value>max_v:
            return max_v
        return value

    def hls_filter(self, image, sens):
        rows,cols,_ = image.shape
        HLS_low = np.array([0,255-sens,0], dtype=np.uint8)
        HLS_high = np.array([255,255,255], dtype=np.uint8)
        inverted = cv2.bitwise_not(cv2.inRange(image, HLS_low, HLS_high))
        return inverted

    def get_float(self, text):
        try:
            res=float(text)
            if '.' not in text:
                res/=10
            return res
        except ValueError:
            return 0

    def plot( self, img):
        if img.shape[1] >1080:
            cv2.imshow('Computer Vision', cv2.resize( img,None,  fx = 1080/img.shape[1], fy = 1080/img.shape[1]))
        else:
            cv2.imshow('Computer Vision', img)

    def plot_lib(self, img):
        if self.axis is None:
            self.figure, self.axis = plt.subplots()
            self.axis.imshow(img)
        else:
            self.axis.imshow(img)
        plt.show()

    def template_match(self, screenshot):
        smeeta_proc_100_processed = self.smeeta_proc_100_processed
        h,w = smeeta_proc_100_processed.shape

        filtered_scan = self.hsv_filter(screenshot, self.icon_color, h_sens=4, s_sens=60, v_scale=0.6)
        #filtered_scan = self.hsv_filter(screenshot, self.icon_color, h_sens=4, s_sens=150, v_scale=0.6)
        #filtered_scan = self.hsv_filter(screenshot, self.icon_color, h_sens=4, s_sens=180, v_scale=0.6)

        match_result = cv2.matchTemplate(filtered_scan, smeeta_proc_100_processed, cv2.TM_CCOEFF_NORMED)
        #match_result = cv2.matchTemplate(~filtered_scan, ~smeeta_proc_100_processed, cv2.TM_CCORR_NORMED, mask=~self.smeeta_proc_100_processed)

        (yCoords, xCoords) = np.where( (match_result > self.template_match_threshold) & (match_result <= 1) )
        rects = []
        probs=[]
        # loop over the starting (x, y)-coordinates again
        for (x, y) in zip(xCoords, yCoords):
            # update our list of rectangles
            rects.append((x, y, x + w, y + h))
            probs.append(match_result[y][x])
        res = non_max_suppression(np.array(rects), probs=probs)
        return res, w, h

    def scan_match_text(self):
        ui_screenshot = self.wincap.get_screenshot(36)
        if ui_screenshot is None: return

        w, h, _ = ui_screenshot.shape
        screenshot_time = time.time()

        filtered_scan = self.text_hsv_filter(ui_screenshot, self.text_color)
        # rotate screenshot to straighten text
        M = cv2.getRotationMatrix2D((w-1,0), self.ui_rotation, 1)
        rotated_filtered_scan = cv2.warpAffine(filtered_scan, M, (h, w), borderMode = cv2.BORDER_CONSTANT, borderValue =255)
        scaled_rotated_filtered_scan = cv2.resize(rotated_filtered_scan,None,  fx = 5, fy = 5, interpolation = cv2.INTER_LANCZOS4)

        output = pytesseract.image_to_data(scaled_rotated_filtered_scan, lang='eng',config='-c tessedit_do_invert=0 -c tessedit_char_whitelist="0123456789x.%m " --psm 11', output_type=pytesseract.Output.DICT)
        for i in range(len(output['level'])):
            proc_time = self.get_float(output['text'][i])

            confident = (float(output['conf'][i]) >= 50)
            valid_width = True
            if len(self.proc_expiry_queue)>0:
                valid_time = (proc_time > self.max_time-10 and proc_time <= self.max_time) and proc_time > ( self.proc_expiry_queue[-1] - screenshot_time + 25 )
            else:
                valid_time = (proc_time > self.max_time-10 and proc_time <= self.max_time)
            if confident and valid_width and valid_time:
                self.procs_active += 1
                self.proc_expiry_queue.append( proc_time + screenshot_time )
                self.proc_warn_queue.append(1)
                self.sound_queue.append('procs_active_%d.mp3'%self.procs_active)
                self.main_window.smeeta_time_reference = screenshot_time-(self.max_time-proc_time)

    def scan_match_template(self):
        ui_screenshot = self.wincap.get_screenshot(36)
        if ui_screenshot is None: return
        w, h, _ = ui_screenshot.shape
        screenshot_time = time.time()
        ui_screenshot_copy = ui_screenshot.copy()

        # get locations of template matches
        locs, wt, ht = self.template_match(ui_screenshot)

        stitched = None
        if len(locs) > 0:
            # filter image
            # text
            filtered_scan = self.text_hsv_filter(ui_screenshot, self.text_color)
            rotated_filtered_scan = cv2.warpAffine(filtered_scan, self.M, (h, w), borderMode = cv2.BORDER_CONSTANT, borderValue =255)

            # convert points to rotated space
            for i in range(len(locs)):
                xn,yn = locs[i][0],locs[i][1]
                rotated_coords= np.matmul( self.M[:,:-1],np.array([[xn],[yn]]) ).astype(int)
                xr,yr = *rotated_coords[0], *rotated_coords[1]
                x1,y1,x2,y2 = (xr-15, yr+ht+3, xr+wt+15, yr+ht+25)
                y0,x0=np.shape(rotated_filtered_scan)
                if y1>y0 or y2>y0 or x1>x0 or x2>x0:
                    #print("Rotated coordinates out of bounds of image")
                    continue
                if stitched is not None:
                    by=abs(abs(y2-y1)-np.shape(stitched)[0])
                    if abs(y2-y1)<np.shape(stitched)[0]:
                        img_with_border = cv2.copyMakeBorder(rotated_filtered_scan[y1:y2,x1:x2], by, 0, 0, 0, cv2.BORDER_CONSTANT, value=[255,255,255])
                    elif abs(y2-y1)>np.shape(stitched)[0]:
                        img_with_border = rotated_filtered_scan[y1:(y2-by),x1:x2]
                    else:
                        img_with_border = rotated_filtered_scan[y1:y2,x1:x2]
                    # add horizontal border to separate detections
                    img_with_border = cv2.copyMakeBorder(img_with_border, 0, 0, 10, 10, cv2.BORDER_CONSTANT, value=[255,255,255])

                    stitched = np.concatenate((stitched, img_with_border), axis=1)
                else:
                    stitched = rotated_filtered_scan[y1:y2,x1:x2]

            if stitched is not None and stitched.shape[0]!=0 and stitched.shape[1]!=0:
                # scale up stitched image
                scaled_rotated_filtered_scan = cv2.resize(stitched, None,  fx = 5, fy = 5, interpolation = cv2.INTER_LANCZOS4)

                blur = cv2.GaussianBlur(scaled_rotated_filtered_scan,(3,3),0)
                blur = cv2.threshold(blur, 10, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

                output = pytesseract.image_to_data(blur, lang='eng',config='-c tessedit_do_invert=0 -c tessedit_char_whitelist="0123456789x.%m " --psm 11', output_type=pytesseract.Output.DICT)
                j=0
                for i in range(len(output['level'])):
                    proc_time = self.get_float(output['text'][i])
                    confident = (float(output['conf'][i]) >= 50)
                    valid_width = True

                    if len(output['level'])>0 and output['text'][i] !='' and float(output['conf'][i]) > 0:
                        if j < len(self.main_window.image_label_list):
                            (xd, yd, wd, hd) = (output['left'][i], output['top'][i], output['width'][i], output['height'][i])
                            if wd>0 and hd>0:
                                # create 3 channel image to display colored rectangle
                                stacked_img = np.stack((blur,)*3, axis=-1)
                                cv2.rectangle(stacked_img, (xd, yd), (xd+wd, yd+hd), (0, 0, 255), 4)
                                detection_image = get_bordered_image( stacked_img, xd, yd, xd+wd, yd+hd )

                                # convert the image to Qt format
                                qt_img = self.convert_cv_qt(detection_image)
                                # display it
                                self.main_window.image_label_list[j].setPixmap(qt_img)
                                self.main_window.label_image_label_list[j].setText("%.1f, Conf: %s"%(proc_time, output['conf'][i]))
                                j+=1

                    if len(self.proc_expiry_queue)>0:
                        valid_time = (proc_time > self.max_time-10 and proc_time <= self.max_time) and proc_time > ( self.proc_expiry_queue[-1] - screenshot_time + 25 )
                    else:
                        valid_time = (proc_time > self.max_time-10 and proc_time <= self.max_time)
                    if confident and valid_width and valid_time:
                        self.procs_active += 1
                        self.proc_expiry_queue.append( proc_time + screenshot_time )
                        self.proc_warn_queue.append(1)
                        self.sound_queue.append('procs_active_%d.mp3'%self.procs_active)
                        date_string = datetime.datetime.fromtimestamp(int(screenshot_time-(self.max_time-proc_time))).strftime('%Y-%m-%d %H:%M:%S')
                        logging.info('Affinity proc detected at time: %s (%d). %d procs active'%(date_string, int(screenshot_time-(self.max_time-proc_time)), self.procs_active))
                        self.append_proc_data( screenshot_time-(self.max_time-proc_time) )
                        self.main_window.smeeta_time_reference = screenshot_time-(self.max_time-proc_time)
                for remain in range(j, len(self.main_window.image_label_list)):
                    self.main_window.image_label_list[remain].clear()
                    self.main_window.label_image_label_list[remain].setText("")

    def get_next_expiry(self):
        self.update_stats()
        if len(self.proc_expiry_queue) == 0:
            return None
        return self.proc_expiry_queue[0]

    def get_procs_active(self):
        self.update_stats()
        return self.procs_active

    def update_stats(self):
        if len(self.proc_expiry_queue)>0:
            if self.proc_expiry_queue[0] - time.time() < 18 and self.proc_warn_queue[0]>0:
                self.proc_warn_queue[0]=0
                self.sound_queue.append('expiry_imminent.mp3')

            if time.time()>=self.proc_expiry_queue[0]:
                self.proc_expiry_queue.popleft()
                self.proc_warn_queue.popleft()
                self.procs_active -= 1

                self.sound_queue.append('procs_active_%d.mp3'%self.procs_active)

    def show_icon_threshold(self):
        ui_screenshot = self.wincap.get_screenshot()
        if ui_screenshot is None: return
        filtered_scan = self.hsv_filter(ui_screenshot, self.icon_color, h_sens=4, s_sens=60, v_scale=0.6)
        self.plot(filtered_scan)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def show_text_threshold(self):
        ui_screenshot = self.wincap.get_screenshot()
        if ui_screenshot is None: return
        filtered_scan = self.text_hsv_filter(ui_screenshot, self.text_color)
        self.plot(filtered_scan)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def append_proc_data(self, val):
        with open('smeeta_history.csv','a+') as fd:
            fd.write('%s\n'%str(val))

    def convert_cv_qt(self, cv_img):
        """Convert from an opencv image to QPixmap"""
        rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        convert_to_Qt_format = QtGui.QImage(rgb_image.data, w, h, bytes_per_line, QtGui.QImage.Format_RGB888)
        p = convert_to_Qt_format.scaled(51, 21, QtCore.Qt.KeepAspectRatio)
        #, QtCore.Qt.KeepAspectRatio
        return QtGui.QPixmap.fromImage(p)

def non_max_suppression(boxes, probs=None, overlapThresh=0):
        # if there are no boxes, return an empty list
        if len(boxes) == 0:
                return []

        # if the bounding boxes are integers, convert them to floats -- this
        # is important since we'll be doing a bunch of divisions
        if boxes.dtype.kind == "i":
                boxes = boxes.astype("float")

        # initialize the list of picked indexes
        pick = []

        # grab the coordinates of the bounding boxes
        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]

        # compute the area of the bounding boxes and grab the indexes to sort
        # (in the case that no probabilities are provided, simply sort on the
        # bottom-left y-coordinate)
        area = (x2 - x1 + 1) * (y2 - y1 + 1)
        idxs = y2

        # if probabilities are provided, sort on them instead
        if probs is not None:
                idxs = probs

        # sort the indexes
        idxs = np.argsort(idxs)

        # keep looping while some indexes still remain in the indexes list
        while len(idxs) > 0:
                # grab the last index in the indexes list and add the index value
                # to the list of picked indexes
                last = len(idxs) - 1
                i = idxs[last]
                pick.append(i)

                # find the largest (x, y) coordinates for the start of the bounding
                # box and the smallest (x, y) coordinates for the end of the bounding
                # box
                xx1 = np.maximum(x1[i], x1[idxs[:last]])
                yy1 = np.maximum(y1[i], y1[idxs[:last]])
                xx2 = np.minimum(x2[i], x2[idxs[:last]])
                yy2 = np.minimum(y2[i], y2[idxs[:last]])

                # compute the width and height of the bounding box
                w = np.maximum(0, xx2 - xx1 + 1)
                h = np.maximum(0, yy2 - yy1 + 1)

                # compute the ratio of overlap
                overlap = (w * h) / area[idxs[:last]]

                # delete all indexes from the index list that have overlap greater
                # than the provided overlap threshold
                idxs = np.delete(idxs, np.concatenate(([last],
                        np.where(overlap > overlapThresh)[0])))

        # return only the bounding boxes that were picked
        return boxes[pick].astype("int")

def get_bordered_image(img, x1,y1,x2,y2, border=10):
    yo,xo,_ = img.shape
    return img[max(0,y1-border):min(yo,y2+border), max(0,x1-border):min(xo,x2+border)]

