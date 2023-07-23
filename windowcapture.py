import numpy as np
import win32gui, win32ui
from ctypes import windll
import time

class WindowCapture:
    hwnd = None
    window_name = None
    cropped_x = cropped_y = 0

    # constructor
    def __init__(self, window_name, capture_size, window):
        self.window = window
        self.window_name = window_name
        # find the handle for the window we want to capture
        self.cap_w, self.cap_h = capture_size
        self.screenshot_timestamp = 0
        self.hwnd = None
        self.find_window()

    def find_window(self):
        self.hwnd = win32gui.FindWindow(None, self.window_name)
        if not self.hwnd:
            print(f'Window not found: {self.window_name}')
            return False
        return True
    
    def is_window(self):
        if self.hwnd is None or not win32gui.IsWindow(self.hwnd):
            if not self.find_window():
                print(f'Window called "{self.window_name}" not found.')
                return False
        return True

    def get_screenshot(self):
        # check if window handle is still valid
        if not self.hwnd or not win32gui.IsWindow(self.hwnd):
            if not self.find_window():
                return None
            
        try:
            left, top, right, bot = win32gui.GetWindowRect(self.hwnd)
        except Exception as e: 
            print(e)
            return None
        
        win_w = right - left
        win_h = bot - top
        # y_border_thickness = GetSystemMetrics(33) + GetSystemMetrics(4)
        y_border_thickness = 0

        #Upper left corner of detection box, given top left of screen is 0,0
        self.cropped_x, self.cropped_y  = ( int(win_w - self.cap_w) , int(y_border_thickness) )

        # get the window image data
        try:
            wDC = win32gui.GetWindowDC(self.hwnd)
        except:
            raise Exception('Window not found: {}'.format(self.window_name))

        dcObj = win32ui.CreateDCFromHandle(wDC)
        cDC = dcObj.CreateCompatibleDC()
        dataBitMap = win32ui.CreateBitmap()
        dataBitMap.CreateCompatibleBitmap(dcObj, win_w, win_h)
        cDC.SelectObject(dataBitMap)
        
        t1=time.time()
        result = windll.user32.PrintWindow(self.hwnd, cDC.GetSafeHdc(), 2)
        self.screenshot_timestamp = (time.time()+t1)/2

        bmpinfo = dataBitMap.GetInfo()

        # convert format
        signedIntsArray = dataBitMap.GetBitmapBits(True)
        img = np.fromstring(signedIntsArray, dtype='uint8')
        img.shape = (bmpinfo['bmHeight'], bmpinfo['bmWidth'], 4)
        img = img[self.cropped_y:self.cropped_y+self.cap_h, self.cropped_x:, :]

        # free resources
        dcObj.DeleteDC()
        cDC.DeleteDC()
        win32gui.ReleaseDC(self.hwnd, wDC)
        win32gui.DeleteObject(dataBitMap.GetHandle())

        return np.ascontiguousarray(img[...,:3])
