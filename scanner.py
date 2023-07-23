# This Python file uses the following encoding: utf-8
import os
import time
from collections import deque

from windowcapture import WindowCapture
from PySide6 import QtGui
from PySide6 import QtCore

import pytesseract
import cv2
import numpy as np
import scipy.signal
import pandas as pd
import matplotlib.pyplot as plt
import constants
from math import ceil, floor
import rc_icons
  
class ScreenScanner:
    def __init__(self, main_window):
        pytesseract.pytesseract.tesseract_cmd = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'Tesseract-OCR\\tesseract.exe')
        self.main_window=main_window
        self.smeeta_history_file = os.path.join(self.main_window.script_folder,'charm_history.csv')

        ui_scale = self.main_window.window_data.ui_scale
        ui_rotation = -3.65/ui_scale
        self.M = cv2.getRotationMatrix2D((0,0), ui_rotation, 1)

        self.affinity_proc_template = downsample_icon(load_from_qrc(f':/icons/{int(round_half_down(ui_scale)*100)}.png'), ui_scale)
        
        self.screen_capture = WindowCapture('Warframe', ( int(750*(1+(ui_scale-1)*0.5)) , int(300*(1+(ui_scale-1)*0.5)) ) , self.main_window)

        # white text requires a special filter
        self.text_hsv_filter = white_hsv_filter if self.main_window.ui.text_color_widget.color_hsv == [0,0,255] else hsv_filter

        self.template_match_status_text = ''
        self.proc_validator = ProcValidator(main_window.window_data.duration_multiplier, self)

        self.affinity_proc_list = []
        self.sound_queue = deque([])

        self.debug_image = None
        self.exit_=False

    def update_ui_scale(self):
        ui_scale = self.main_window.window_data.ui_scale
        self.screen_capture = WindowCapture('Warframe', ( int(750*(1+(ui_scale-1)*0.5)) , int(300*(1+(ui_scale-1)*0.5)) ) , self.main_window)
        ui_rotation = -3.65/self.main_window.window_data.ui_scale
        self.M = cv2.getRotationMatrix2D((0,0), ui_rotation, 1)
        self.affinity_proc_template = downsample_icon(load_from_qrc(f':/icons/{int(round_half_down(ui_scale)*100)}.png'), ui_scale)

    def update_text_color(self):
        self.text_hsv_filter = white_hsv_filter if self.main_window.ui.text_color_widget.color_hsv == [0,0,255] else hsv_filter

    def reset(self):
        self.affinity_proc_list = []
        self.sound_queue = deque([])

    def get_search_areas(self, img_inv):
        result = cv2.bitwise_not(img_inv.copy())
        contours = cv2.findContours(result, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = contours[0] if len(contours) == 2 else contours[1]
        contour_list = []
        for cntr in contours:
            x,y,w,h = cv2.boundingRect(cntr)
            contour_list.append((x,y,w,h))
        return contour_list

    def template_match(self, screenshot, plot=False):
        icon_color = self.main_window.ui.smeeta_icon_widget.color_hsv

        template = self.affinity_proc_template
        h,w = template.shape

        filtered_scan = hsv_filter(screenshot, icon_color, h_sens=4, s_sens=60, v_scale=0.6)
        match_result = np.zeros_like(filtered_scan).astype(np.float32)
        
        interesting_area_coords = self.get_search_areas(filtered_scan)

        input_img = filtered_scan.copy().astype(np.float32) 
        template_mean = template.mean()
        template_img_norm = template.copy().astype(np.float32) - template_mean

        debug_image = filtered_scan.copy()

        for xa,ya,wa,ha in interesting_area_coords:
            detect_area = np.sum(255-input_img[ya:ya+ha, xa:xa+wa])
            template_area = np.sum(255-template)
            
            if detect_area > template_area*0.5 and detect_area < template_area*2:
                xs,ys,xe,ye = get_padded_coordinates(xa,ya,wa,ha, w, h, input_img.shape[1], input_img.shape[0])
                interesting_area = input_img[ys:ye, xs:xe]
                interesting_area_mean = interesting_area.mean()
                interesting_area_norm = interesting_area - interesting_area_mean

                # set fillvalue to 0 or interesting_area_norm.min()
                # the larger image should always be used as in1
                res = scipy.signal.correlate2d(interesting_area_norm, template_img_norm, mode='same', fillvalue=0)
                
                # get index where overlap is maximum
                corr_max_index = res.argmax()
                sel_y, sel_x = np.unravel_index(corr_max_index, res.shape)

                denom = np.sqrt(abs(np.sum(interesting_area_norm ** 2) * np.sum(template_img_norm ** 2)))
                if denom == 0:
                    continue
                normalized_peak = res[sel_y, sel_x] / denom

                # the coordinate of the max is the center of the template - to get the top left corner subtract the width and height of the template
                ymin = max(0, sel_y-floor(h/2))+ys
                xmin = max(0, sel_x-floor(w/2))+xs
                match_result[ymin,xmin] = normalized_peak

        template_match_threshold = self.main_window.window_data.template_match_threshold
        # get locations where the template matches more than the required percent
        (yCoords, xCoords) = np.where( (match_result > template_match_threshold) & (match_result <= 1) )
        self.template_match_status_text = f'Icon match: {np.max(match_result[match_result<=1])*100:.1f}%'
        zipped = list(zip(xCoords, yCoords))

        if plot == True:
            for x,y in zipped:
                cv2.rectangle(debug_image, (x,y), (x+w,y+h), (0, 0, 255), 1)
                debug_image[y:y+h,x:x+w] = debug_image[y:y+h,x:x+w]*0.5 + template*0.5
            self.debug_image = debug_image

        return zipped, w, h

    def find_charm_proc(self):
        text_color = self.main_window.ui.text_color_widget.color_hsv
        ui_scale = self.main_window.window_data.ui_scale
        # update existing procs
        if len(self.affinity_proc_list) > 0:
            current_time_unix = time.time()
            # issue expiry warning
            next_expiry = self.get_next_expiry_unix()
            if next_expiry is not None:
                if next_expiry - current_time_unix < 18 and not self.affinity_proc_list[0].expiry_warning_issued:
                    self.affinity_proc_list[0].expiry_warning_issued = True
                    self.sound_queue.append('expiry_imminent.mp3')
                # remove expired procs
                if current_time_unix >= next_expiry:
                    self.affinity_proc_list.pop(0)
                    self.sound_queue.append('procs_active_%d.mp3'%(len(self.affinity_proc_list)))

        ui_screenshot = self.screen_capture.get_screenshot()
        screenshot_unix = self.screen_capture.screenshot_timestamp

        if ui_screenshot is None: 
            self.exit_ = True
            print("Error: Cannot find Warframe window")
            return
        w, h, _ = ui_screenshot.shape

        # get locations of template matches
        locs, wt, ht = self.template_match(ui_screenshot, plot=False)

        stitched = None
        if len(locs) > 0:
            filtered_scan = self.text_hsv_filter(ui_screenshot, text_color)
            rotated_filtered_scan = cv2.warpAffine(filtered_scan, self.M, (h, w), borderMode = cv2.BORDER_CONSTANT, borderValue=255)

            # convert points to rotated space
            for i in range(len(locs)):
                xn,yn = locs[i][0],locs[i][1]
                rotated_coords= np.matmul( self.M[:,:-1],np.array([[xn],[yn]]) ).astype(int)
                xr,yr = *rotated_coords[0], *rotated_coords[1]
                x1,y1,x2,y2 = (int(xr-15*ui_scale), int(yr+ht+3*ui_scale), int(xr+wt+15.5*ui_scale), int(yr+ht+25.5*ui_scale))
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
                scaled_rotated_filtered_scan = cv2.resize(stitched, None,  fx = max(1, int(5/ui_scale)), fy = max(1,int(5/ui_scale)), interpolation = cv2.INTER_LANCZOS4)

                blur = cv2.GaussianBlur(scaled_rotated_filtered_scan,(3,3),0)
                blur = cv2.threshold(blur, 10, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

                output = pytesseract.image_to_data(blur, lang='eng',config='-c tessedit_do_invert=0 -c tessedit_char_whitelist="0123456789x.%m " --psm 11', output_type=pytesseract.Output.DICT)
                j=0
                for i in range(len(output['level'])):
                    proc_time = text2float(output['text'][i])
                    confident = (float(output['conf'][i]) >= 50)

                    if len(output['level'])>0 and output['text'][i] !='' and float(output['conf'][i]) > 0:
                        if j < self.main_window.window_data.scan_display_data.max_images:
                            (xd, yd, wd, hd) = (output['left'][i], output['top'][i], output['width'][i], output['height'][i])
                            if wd>0 and hd>0:
                                # create 3 channel image to display colored rectangle
                                stacked_img = np.stack((blur,)*3, axis=-1)
                                cv2.rectangle(stacked_img, (xd, yd), (xd+wd, yd+hd), (0, 0, 255), 4)
                                detection_image = get_bordered_image( stacked_img, xd, yd, xd+wd, yd+hd )
                                # convert the image to Qt format
                                qt_img = convert_cv_qpixmap(detection_image, 51, 21)
                                self.main_window.window_data.scan_display_data.add_element(qt_img, f'{proc_time:.1f}, Conf: {output["conf"][i]}')
                                j+=1

                    if confident:
                        self.proc_validator.process_detection(proc_time, screenshot_unix)              

    def get_next_expiry_unix(self):
        if len(self.affinity_proc_list)>0:
            return self.affinity_proc_list[0].get_expiry_unix_s()

class AffinityProc():
    def __init__(self, start_time_unix_s, duration) -> None:
        self.duration = duration
        self.start_time_unix_s =  start_time_unix_s
        self.expiry_warning_issued = False

    def get_expiry_unix_s(self):
        expiry_unix_s = self.start_time_unix_s + self.duration 
        return expiry_unix_s

class ProcValidator():
    def __init__(self, duration_multiplier, scanner:ScreenScanner) -> None:
        self.scanner = scanner
        self.proc_list = []
        self.duration_multiplier = duration_multiplier
        self.proc_durations = np.array([v for _,v in constants.PROCNAME_DURATION.items()]) * duration_multiplier
        self.min_time_restrictions = [self.proc_durations[0]-self.proc_durations[1], self.proc_durations[1]-self.proc_durations[2], 0]
        self.proc_names = list(constants.PROCNAME_DURATION)
        self.last_proc_start_unix_s = 0
    
    def process_detection(self, duration_remaining, screenshot_unix):
        expiry_timestamp_unix_s = screenshot_unix + duration_remaining

        if len(self.proc_list) > 0:
            diffs = [abs(expiry_timestamp_unix_s - proc.expiry_timestamp_unix_s) if abs(proc.validation_cycle-screenshot_unix) > 1e-5 else 100000 for proc in self.proc_list]
            mindex = np.argmin(diffs)
            if diffs[mindex] < 0.25: # this is a proc we already found
                # set the validation cycle so that we don't validate the same proc more than once in the same screenshot
                self.proc_list[mindex].validation_cycle = screenshot_unix

                # proc is validated, do not need to add more validations
                if self.proc_list[mindex].validated:
                    return
                
                # proc that exists but is not validated
                self.proc_list[mindex].add_validation()
                if self.proc_list[mindex].validated:
                    self.last_proc_start_unix_s = max(self.last_proc_start_unix_s, self.proc_list[mindex].start_timestamp_unix_s)
                    # supress other procs within 26 seconds of this one - careful, this right part will remove the proc itself from the list 
                    #self.proc_list = [self.proc_list[mindex]] + [e for e in self.proc_list if (abs(e.start_timestamp_unix_s - self.proc_list[mindex].start_timestamp_unix_s) > 26)]
                    self.proc_list = [e for e in self.proc_list if ( (abs(e.start_timestamp_unix_s - self.proc_list[mindex].start_timestamp_unix_s) > 26) or (e is self.proc_list[mindex]))]
                return

        # new proc - find the closest match
        diff = self.proc_durations - duration_remaining
        non_negative_indices = np.where(diff >= 0)[0]
        if len(non_negative_indices) == 0:
            return
        min_index = non_negative_indices[np.argmin(diff[non_negative_indices])]

        valid_time_restriction = abs(screenshot_unix - self.last_proc_start_unix_s)>26 and duration_remaining > self.min_time_restrictions[min_index]

        if valid_time_restriction:
            self.proc_list.append(self.Proc(self.proc_names[min_index], self.proc_durations[min_index], duration_remaining, screenshot_unix, self.scanner))

    def remove_expired_procs(self):
        if len(self.proc_list) == 0:
            return
        cur_time = time.time()
        self.proc_list = [e for e in self.proc_list if e.expiry_timestamp_unix_s > cur_time]

    class Proc():
        def __init__(self, name, base_duration, expiry_s, screenshot_unix, scanner:ScreenScanner) -> None:
            self.name = name
            self.base_duration = base_duration
            self.expiry_s = expiry_s
            self.detection_timestamp_unix_s = screenshot_unix
            self.expiry_timestamp_unix_s = self.detection_timestamp_unix_s + expiry_s
            self.start_timestamp_unix_s = self.detection_timestamp_unix_s - (base_duration - expiry_s)
            self.validations = 1
            self.scanner = scanner
            self.validated = False
            self.scan_start_timestamp_unix_s = max(self.scanner.main_window.monitor.log_parser.mission_start_timestamp_unix_s, self.scanner.main_window.monitor.scan_start_timestamp_unix_s)
            self.validation_cycle = screenshot_unix

        def add_validation(self):
            self.validations += 1
            if self.validations >= 3 and not self.validated:
                self.validated = True
                data = {"name":self.name ,"proc_start_timestamp_unix_s":self.start_timestamp_unix_s, "scan_start_timestamp_unix_s": self.scan_start_timestamp_unix_s, "proc_duration_s":self.base_duration,
                            "detection_value":self.expiry_s, "detection_timestamp_unix_s": self.detection_timestamp_unix_s, "proc_expiry_timestamp_unix_s":self.expiry_timestamp_unix_s}
                pd.DataFrame([data]).to_csv(self.scanner.smeeta_history_file, mode='a', header=not os.path.isfile(self.scanner.smeeta_history_file), index=False)

                if self.name == "Affinity":
                    affinity_duration = self.scanner.main_window.window_data.affinity_proc_duration
                    self.scanner.affinity_proc_list.append( AffinityProc(self.start_timestamp_unix_s, affinity_duration) )
                    self.scanner.sound_queue.append('procs_active_%d.mp3'%(len(self.scanner.affinity_proc_list)))
                if self.scanner.main_window.window_data.play_all_proc_sounds:
                    if self.name == "Energy Refund":
                        self.scanner.sound_queue.append('energy_proc.mp3')
                    elif self.name == "Critical Chance":
                        self.scanner.sound_queue.append('critical_proc.mp3')

                self.scanner.main_window.charm_history.new_proc(self)

def get_padded_coordinates(x, y, w, h, w2, h2, xmax, ymax):
    x1,y1,x2,y2 = x,y,x+w,y+h
    if w < w2:
        pad = (w2 - w)/2+1
        x1 = max(0, x1-pad)
        x2 = min(xmax, x2+pad)
    if h < h2:
        pad = (h2 - h)/2+1
        y1 = max(0, y1-pad)
        y2 = min(ymax, y2+pad)
    return int(x1), int(y1), int(x2), int(y2)

def get_bordered_image(img, x1,y1,x2,y2, border=10):
    yo,xo,_ = img.shape
    return img[max(0,y1-border):min(yo,y2+border), max(0,x1-border):min(xo,x2+border)]

def text2float(text):
    try:
        res=float(text)
        if '.' not in text:
            res/=10
        return res
    except ValueError:
        return 0

def convert_cv_qpixmap(cv_img, w_s=None, h_s=None):
    """Convert from an opencv image to QPixmap"""
    rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
    h, w, ch = rgb_image.shape
    bytes_per_line = ch * w
    qimage = QtGui.QImage(rgb_image.data, w, h, bytes_per_line, QtGui.QImage.Format_RGB888)
    if w_s is not None and  h_s is not None:
        qimage = qimage.scaled(w_s, h_s, QtCore.Qt.KeepAspectRatio)
    return QtGui.QPixmap.fromImage(qimage)

def load_from_qrc(qrc, flag=cv2.IMREAD_COLOR):
    file = QtCore.QFile(qrc)
    m = None
    if file.open(QtCore.QIODevice.ReadOnly):
        sz = file.size()
        buf = np.frombuffer(file.read(sz), dtype=np.uint8)
        m = cv2.imdecode(buf, flag)
    return m

def limit_color(value, max_v=255):
    if value <0:
        return 0
    elif value>max_v:
        return max_v
    return value

# Hue range is [0,179], Saturation range is [0,255] and Value range is [0,255].
def hsv_filter(image, color, h_sens=5, s_sens=40, v_scale=0.3):
    image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    HSV_low = np.array([limit_color(color[0]-h_sens,179), limit_color(color[1]-s_sens), limit_color(color[2]*v_scale)], dtype=np.uint8)
    #HSV_high = np.array([limit_color(color[0]+h_sens), limit_color(color[1]+1), limit_color(color[2]+40)], dtype=np.uint8)
    HSV_high = np.array([limit_color(color[0]+h_sens), 255, 255], dtype=np.uint8)
    inverted = cv2.bitwise_not(cv2.inRange(image_hsv, HSV_low, HSV_high))
    return inverted

def white_hsv_filter(image, color, h_sens=5, s_sens=40, v_scale=0.5):
    image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    HSV_low = np.array([0, 0, 150], dtype=np.uint8)
    HSV_high = np.array([179, 26, 255], dtype=np.uint8)
    inverted = cv2.inRange(image_hsv, HSV_low, HSV_high)
    return cv2.bitwise_not(inverted)

def hls_filter(image, sens):
    rows,cols,_ = image.shape
    HLS_low = np.array([0,255-sens,0], dtype=np.uint8)
    HLS_high = np.array([255,255,255], dtype=np.uint8)
    inverted = cv2.bitwise_not(cv2.inRange(image, HLS_low, HLS_high))
    return inverted

def concat_images(imga, imgb):
    """
    Combines two color image ndarrays side-by-side.
    """
    ha,wa = imga.shape[:2]
    hb,wb = imgb.shape[:2]
    max_height = np.max([ha, hb])
    total_width = wa+wb
    new_img = np.zeros(shape=(max_height, total_width))
    new_img[:ha,:wa]=imga
    new_img[:hb,wa:wa+wb]=imgb
    return new_img

def round_half_up(n):
    if n % 0.5 == 0:
        return n
    return ceil(n * 2) / 2

def round_half_down(n):
    if n % 0.5 == 0:
        return n
    return floor(n * 2) / 2

def downsample_icon(image_bgr, ui_scale):
    scaled_image = cv2.resize(image_bgr, None, fx=ui_scale/round_half_down(ui_scale), fy=ui_scale/round_half_down(ui_scale))
    _, thresh = cv2.threshold(scaled_image,127,255,cv2.THRESH_BINARY)
    grs_thresh = thresh
    if image_bgr.shape[2]>1:
        grs_thresh = cv2.cvtColor(thresh, cv2.COLOR_BGR2GRAY)

    contours = cv2.findContours(cv2.bitwise_not(grs_thresh), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = contours[0] if len(contours) == 2 else contours[1]

    # find the contour with the largest area
    contour_list = []
    area_list = []
    for cntr in contours:
        x,y,w,h = cv2.boundingRect(cntr)
        contour_list.append((x,y,w,h))
        area_list.append(w*h)

    x,y,w,h = contour_list[np.argmax(area_list)]
    cropped_image = thresh[y:y+h, x:x+w, 0]
    return cropped_image
    