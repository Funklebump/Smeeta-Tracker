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
import logging
import datetime
import scipy.signal
import pandas as pd
import json
import matplotlib.pyplot as plt
from pathlib import Path
  
class ScreenScanner:
    def __init__(self, main_window):
        pytesseract.pytesseract.tesseract_cmd = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'Tesseract-OCR\\tesseract.exe')
        self.main_window=main_window

        self.smeeta_history_file = os.path.join(self.main_window.script_folder,'smeeta_history.csv')
        if not os.path.isfile(self.smeeta_history_file):
            with open(self.smeeta_history_file, "w") as emptycsv:
                pass

        self.ui_scale = main_window.ui_scale
        self.ui_rotation = -3.65/self.ui_scale
        #self.ui_rotation = -3.5/self.ui_scale

        self.scan_scale = 5/self.ui_scale
        
        with open(os.path.join(main_window.script_folder,'config.json')) as json_file:
            data = json.load(json_file)

        self.text_color = data['text_color_hsv']
        self.icon_color = data['icon_color_hsv']

        self.hue_min = 0
        self.hue_max = 179
        self.saturation_min=0
        self.saturation_max=255
        self.value_max = 255
        self.value_min=0

        self.custom_filter = False

        ri, gi, bi = cv2.cvtColor(np.uint8([[[self.icon_color[0], self.icon_color[1], self.icon_color[2]]]]), cv2.COLOR_HSV2RGB)[0][0]
        self.update_hsv_range(ri, gi, bi, a_min=0.6)

        self.affinity_duration = main_window.affinity_proc_duration
        self.template_match_threshold = main_window.template_match_threshold

        self.paused = False
        self.figure, self.axis = None,None

        self.M = cv2.getRotationMatrix2D((0,0), self.ui_rotation, 1)

        self.affinity_proc_template = cv2.imread(os.path.join(os.path.dirname(os.path.abspath(__file__)),'Templates',f'{int(self.ui_scale*100)}p_new1.png'), 0 )

        self.wincap = WindowCapture('Warframe', ( int(750*(1+(self.ui_scale-1)*0.5)) , int(300*(1+(self.ui_scale-1)*0.5)) ) , self.main_window)

        # white text requires a special filter
        self.text_hsv_filter = white_hsv_filter if self.text_color == [0,0,255] else hsv_filter

        self.template_match_status_text = ''

        self.previous_proc_trigger_timestamp_unix_s = 0
        self.refresh_rate_s = 2

        self.affinity_proc_list = []
        self.sound_queue = deque([])

    def update_ui_settings(self):
        self.ui_scale = self.main_window.ui_scale
        self.ui_rotation = -3.65/self.ui_scale
        self.scan_scale = 5/self.ui_scale
        self.text_color = self.main_window.text_color_hsv
        self.icon_color = self.main_window.icon_color_hsv
        self.text_hsv_filter = white_hsv_filter if self.text_color == [0,0,255] else hsv_filter
        self.wincap = WindowCapture('Warframe', ( int(750*(1+(self.ui_scale-1)*0.5)) , int(300*(1+(self.ui_scale-1)*0.5)) ) , self.main_window)
        self.affinity_proc_template = cv2.imread(os.path.join(os.path.dirname(os.path.abspath(__file__)),'Templates',f'{int(self.ui_scale*100)}p_new1.png'), 0 )

    def reset(self):
        self.affinity_proc_list = []
        self.sound_queue = deque([])
        self.paused = False

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
        template = self.affinity_proc_template
        h,w = template.shape

        filtered_scan = self.icon_filter(screenshot, self.icon_color, h_sens=4, s_sens=60, v_scale=0.6)
        match_result = np.zeros_like(filtered_scan).astype(np.float32)
        
        interesting_area_coords = self.get_search_areas(filtered_scan)

        input_img = filtered_scan.copy().astype(np.float32) 
        template_mean = template.mean()
        template_img_norm = template.copy().astype(np.float32) - template_mean

        for xa,ya,wa,ha in interesting_area_coords:
            detect_area = np.sum(255-input_img[ya:ya+ha, xa:xa+wa])
            template_area = np.sum(255-template)
            
            if detect_area > template_area*0.5 and detect_area < template_area*2 :# and wa<w and ha<h :

                interesting_area = input_img[ya:ya+ha, xa:xa+wa]
                interesting_area_mean = interesting_area.mean()
                interesting_area_norm = interesting_area - interesting_area_mean

                # set fillvalue to 0 or interesting_area_norm.min()
                res = scipy.signal.correlate2d(template_img_norm, interesting_area_norm, mode='same', fillvalue=0)
                
                # get index where overlap is maximum
                corr_max_index = res.argmax()
                sel_y, sel_x = np.unravel_index(corr_max_index, res.shape)

                denom = np.sqrt(abs(np.sum(interesting_area_norm ** 2) * np.sum(template_img_norm ** 2)))
                if denom == 0:
                    continue
                normalized_peak = res[sel_y, sel_x] / denom
                percentage_match = (normalized_peak)*100

                if plot == True:
                    fig, (ax_orig, ax_template, ax_corr) = plt.subplots(3, 1, figsize=(6, 15))
                    ax_orig.imshow(template, cmap='gray')
                    ax_orig.set_title('Original')
                    ax_orig.set_axis_off()
                    ax_template.imshow(interesting_area, cmap='gray')
                    ax_template.set_title('Template')
                    ax_template.set_axis_off()
                    ax_corr.imshow(res, cmap='gray')
                    ax_corr.set_title(f'Cross-correlation ({percentage_match:.1f}%)')
                    ax_corr.set_axis_off()
                    ax_orig.plot(sel_x, sel_y, 'ro')
                    fig.show()

                # use for making templates
                # if percentage_match > 40:
                #     cv2.imwrite(os.path.join(Path(__file__).parent.absolute(), 'Templates', f'{int(100*self.ui_scale)}p_new_t1.png'), interesting_area)
                #     input()

                ymin, ymax = max(0, sel_y-h//2)+ya, max(sel_y+h//2, h)+ya
                xmin, xmax = max(0, sel_x-w//2)+xa, max(sel_x+w//2, w)+xa
                match_result[ymin,xmin] = normalized_peak
            # else:
            #     if plot:
            #         print(detect_area > template_area*0.5 , detect_area < template_area*2 , wa>0.5*w , ha>0.5*h)

        (yCoords, xCoords) = np.where( (match_result > self.template_match_threshold) & (match_result <= 1) )
        self.template_match_status_text = f'Max template match: {np.max(match_result[match_result<=1])*100:.1f}%'
        # if len(yCoords)==0:
        #     print(f'Template match threshold of {self.template_match_threshold*100:.1f}% not satisfied by any elements. Max match found: {np.max(match_result[match_result<=1])*100:.1f}%')

        return list(zip(xCoords, yCoords)), w, h

    def scan_match_template(self):
        # update existing procs
        if len(self.affinity_proc_list) > 0 and not self.paused:
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

        ui_screenshot = self.wincap.get_screenshot()
        if ui_screenshot is None: 
            print("Error: Cannot find Warframe window")
            return
        w, h, _ = ui_screenshot.shape
        screenshot_time_unix = time.time()

        # get locations of template matches
        locs, wt, ht = self.template_match(ui_screenshot)

        stitched = None
        if len(locs) > 0:
            filtered_scan = self.text_hsv_filter(ui_screenshot, self.text_color)
            rotated_filtered_scan = cv2.warpAffine(filtered_scan, self.M, (h, w), borderMode = cv2.BORDER_CONSTANT, borderValue=255)

            # convert points to rotated space
            for i in range(len(locs)):
                xn,yn = locs[i][0],locs[i][1]
                rotated_coords= np.matmul( self.M[:,:-1],np.array([[xn],[yn]]) ).astype(int)
                xr,yr = *rotated_coords[0], *rotated_coords[1]
                x1,y1,x2,y2 = (int(xr-15*self.ui_scale), int(yr+ht+3*self.ui_scale), int(xr+wt+15.5*self.ui_scale), int(yr+ht+25.5*self.ui_scale))
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
                scaled_rotated_filtered_scan = cv2.resize(stitched, None,  fx = max(1, int(5/self.ui_scale)), fy = max(1,int(5/self.ui_scale)), interpolation = cv2.INTER_LANCZOS4)

                blur = cv2.GaussianBlur(scaled_rotated_filtered_scan,(3,3),0)
                blur = cv2.threshold(blur, 10, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

                output = pytesseract.image_to_data(blur, lang='eng',config='-c tessedit_do_invert=0 -c tessedit_char_whitelist="0123456789x.%m " --psm 11', output_type=pytesseract.Output.DICT)
                j=0
                for i in range(len(output['level'])):
                    proc_time = text2float(output['text'][i])
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
                                qt_img = convert_cv_qt(detection_image)
                                self.main_window.add_detection_info(qt_img, f'{proc_time:.1f}, Conf: {output["conf"][i]}')
                                #self.main_window.image_list.append(qt_img)
                                #self.main_window.image_text_list.append(f'{proc_time:.1f}, Conf: {output["conf"][i]}')
                                j+=1

                    if len(self.affinity_proc_list)>0:
                        valid_time = (proc_time > self.affinity_duration-25 and proc_time <= self.affinity_duration) and proc_time > ( self.affinity_proc_list[-1].get_expiry_unix_s() - screenshot_time_unix + 25 )
                    else:
                        valid_time = (proc_time > self.affinity_duration-25 and proc_time <= self.affinity_duration) #proc_time > self.affinity_duration-10 and #TODO
                    #print(f'confident: {confident}, valid time: {valid_time}')
                    if confident and valid_width and valid_time:
                        self.previous_proc_trigger_timestamp_unix_s = proc_time+screenshot_time_unix-self.affinity_duration
                        self.affinity_proc_list.append( AffinityProc(proc_time+screenshot_time_unix-self.affinity_duration, self.affinity_duration) )
                        self.sound_queue.append('procs_active_%d.mp3'%(len(self.affinity_proc_list)))
                        date_string = datetime.datetime.fromtimestamp(int(screenshot_time_unix-(self.affinity_duration-proc_time))).strftime('%Y-%m-%d %H:%M:%S')
                        logging.info('Affinity proc detected at time: %s (%d). %d procs active'%(date_string, int(screenshot_time_unix-(self.affinity_duration-proc_time)), len(self.affinity_proc_list)))
                        pd.DataFrame([{'smeeta_proc_unix_s':screenshot_time_unix-(self.affinity_duration-proc_time)}]).to_csv(self.smeeta_history_file, mode='a', header=not os.path.isfile(self.smeeta_history_file), index=False)

    def get_next_expiry_unix(self):
        if len(self.affinity_proc_list)>0:
            return self.affinity_proc_list[0].get_expiry_unix_s()

    def activate_pause(self, pause_start_unix):
        print("activate pause")
        self.paused = True
        for affinity in self.affinity_proc_list:
            affinity.set_pause_start_time(pause_start_unix)
    
    def deactivate_pause(self, pause_end_unix):
        self.paused = False
        for affinity in self.affinity_proc_list:
            affinity.set_pause_end_time(pause_end_unix)

    # a_min defines the amount of transparency that a pixel has because of interpolation
    def update_hsv_range(self, ri, gi, bi, a_min):
        # a_max is a transparency that is always there
        a_max=0.9
        ri=min(ri,255)
        gi=min(gi,255)
        bi=min(bi,255)
        r1, r2, r3, r4 = ri*a_max+0*(1-a_max), ri*a_max+255*(1-a_max), ri*a_min+0*(1-a_min), ri*a_min+255*(1-a_min)
        g1, g2, g3, g4 = gi*a_max+0*(1-a_max), gi*a_max+255*(1-a_max), gi*a_min+0*(1-a_min), gi*a_min+255*(1-a_min)
        b1, b2, b3, b4 = bi*a_max+0*(1-a_max), bi*a_max+255*(1-a_max), bi*a_min+0*(1-a_min), bi*a_min+255*(1-a_min)

        hl, sl, vl = [],[],[]

        for rl in range(int(min(r1,r2,r3,r4)), int(max(r1,r2,r3,r4))+1):
            for gl in range(int(min(g1,g2,g3,g4)), int(max(g1,g2,g3,g4))+1):
                for bl in range(int(min(b1, b2, b3, b4)), int(max(b1, b2, b3, b4))+1):
                    r=min(rl,255)
                    g=min(gl,255)
                    b=min(bl,255)
                    r = r/255
                    g = g/255
                    b = b/255
                    cmax=max(r,g,b)
                    cmin=min(r,g,b)
                    delta=cmax-cmin
                    if delta==0:
                        h=0
                    elif r==cmax:
                        h=30*((g-b)/delta)%6
                    elif g==cmax:
                        h=30*((b-r)/delta+2)
                    elif b==cmax:
                        h=30*((r-g)/delta+4)
                    if cmax==0:
                        s=0
                    else:
                        s=delta/cmax
                    v=cmax

                    hl.append(h)
                    sl.append(s*255)
                    vl.append(v*255)

        self.hue_min = int(min(hl))
        self.hue_max = min(255, int(max(hl))+1)
        self.saturation_min = int(min(sl))
        self.saturation_max = min(255, int(max(sl))+1)
        self.value_min = int(min(vl))
        self.value_max = min(255, int(max(vl))+1)
    
    # Hue range is [0,179], Saturation range is [0,255] and Value range is [0,255].
    def icon_hsv_filter(self, image):
        image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        HSV_low = np.array([self.hue_min, self.saturation_min, self.value_min], dtype=np.uint8)
        HSV_high = np.array([self.hue_max, self.saturation_max, self.value_max], dtype=np.uint8)
        inverted = cv2.bitwise_not(cv2.inRange(image_hsv, HSV_low, HSV_high))
        return inverted

    def icon_filter(self, screenshot, color, h_sens=4, s_sens=60, v_scale=0.6):
        if self.custom_filter:
            return self.icon_hsv_filter(screenshot)
        else:
            return hsv_filter(screenshot, color, h_sens=4, s_sens=60, v_scale=0.6)

    def display_detection_area(self):
        ui_screenshot = self.wincap.get_screenshot()
        if ui_screenshot is None: 
            print("Error: Cannot find Warframe window")
            return
        cv2.imshow("display_detection_area", ui_screenshot)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    def display_icon_filter(self):
        ui_screenshot = self.wincap.get_screenshot()
        if ui_screenshot is None: 
            print("Error: Cannot find Warframe window")
            return
        filtered_scan = self.icon_filter(ui_screenshot, self.icon_color, h_sens=4, s_sens=60, v_scale=0.6)
        cv2.imshow("display_icon_filter", filtered_scan)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def display_text_filter(self):
        ui_screenshot = self.wincap.get_screenshot()
        if ui_screenshot is None: 
            print("Error: Cannot find Warframe window")
            return
        filtered_scan = self.text_hsv_filter(ui_screenshot, self.text_color)
        cv2.imshow("display_text_filter", filtered_scan)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

class AffinityProc():
    def __init__(self, start_time_unix_s, duration) -> None:
        self.duration = duration
        self.start_time_unix_s =  start_time_unix_s
        self.paused_time_s = 0
        self.pause_start_time_unix_s =  None
        self.pause_end_time_unix_s =  None
        self.expiry_warning_issued = False

    def get_expiry_unix_s(self):
        if self.pause_start_time_unix_s is not None and self.pause_end_time_unix_s is not None:
            self.paused_time_s = self.pause_end_time_unix_s - self.pause_start_time_unix_s
        expiry_unix_s = self.start_time_unix_s + self.duration + self.paused_time_s 
        return expiry_unix_s

    def set_pause_start_time(self, pause_start_unix):
        self.pause_start_time_unix_s = pause_start_unix

    def set_pause_end_time(self, pause_end_unix):
        self.pause_end_time_unix_s = pause_end_unix
        # self.paused_time_s += pause_end_unix - self.pause_start_time_unix_s

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

def convert_cv_qt(cv_img):
    """Convert from an opencv image to QPixmap"""
    rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
    h, w, ch = rgb_image.shape
    bytes_per_line = ch * w
    convert_to_Qt_format = QtGui.QImage(rgb_image.data, w, h, bytes_per_line, QtGui.QImage.Format_RGB888)
    p = convert_to_Qt_format.scaled(51, 21, QtCore.Qt.KeepAspectRatio)
    return QtGui.QPixmap.fromImage(p)

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