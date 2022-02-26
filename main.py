# This Python file uses the following encoding: utf-8
import os
from pathlib import Path
import sys
import threading
import time
import json

from win32gui import GetWindowText, GetForegroundWindow

from PySide6.QtWidgets import QApplication, QWidget
from PySide6 import QtWidgets
from PySide6 import QtGui
from PySide6 import QtCore
from PySide6.QtCore import QFile
from PySide6.QtUiTools import QUiLoader

from scanner import Scanner
from playsound import playsound
from windowcapture import WindowCapture
from EeLogParser import EeLogParser
import cv2
from matplotlib import pyplot as plt
import logging
import datetime
import ctypes
import requests

SMEETA_ICON_COLOR_TYPE = 42
TEXT_COLOR_TYPE = 42+1
version_link = "https://raw.github.com/A-DYB/smeeta-tracker-2/main/version.json"

print( requests.get(version_link).text)
y = json.loads(requests.get(version_link).text)

class MainWindow(QWidget):
    def __init__(self):
        super(MainWindow, self).__init__()
        self.ui=None
        self.load_ui()
        self.dirname = os.path.dirname(os.path.abspath(__file__))
        self.clear = lambda: os.system('cls')
        user32 = ctypes.windll.user32
        self.screencap = WindowCapture('Warframe', ( user32.GetSystemMetrics(0) , user32.GetSystemMetrics(1) ), self.ui )
        self.screenshot = None
        self.screenshot_hsv = None
        self.color_event = TEXT_COLOR_TYPE
        self.icon_color_hsv = [95, 255, 255]
        self.text_color_hsv = [0, 0, 255]

        self.latest_version_json = json.loads(requests.get(version_link).text)
        with open("version.json") as f:
            self.current_version_json = json.load(f)
        if self.latest_version_json["version"] != self.current_version_json["version"]:
            self.ui.version_label.setText('''<a href='https://github.com/A-DYB/smeeta-tracker-2/releases'>Updates available!</a>''')
            self.ui.version_label.setOpenExternalLinks(True)
        else:
            self.ui.version_label.setText("Version:%s - No updates available"%self.current_version_json["version"])

        self.keep_threads_alive = False

        self.dialog_thread_active = False
        self.scan_thread_active = False
        self.sound_thread_active = False
        self.thread_list=[]

        self.dialog = None
        self.scanner = None
        self.sounds = False

        app.aboutToQuit.connect(self.closeEvent)

        self.ui.overlay_checkbox.stateChanged.connect(self.toggle_overlay)
        self.ui.sounds_checkbox.stateChanged.connect(self.toggle_sounds)
        self.ui.start_button.clicked.connect(self.start_scanning)
        self.ui.stop_button.clicked.connect(self.stop_scanning)

        self.ui.test_text_threshold_button.clicked.connect(self.show_text_threshold)
        self.ui.test_icon_threshold_button.clicked.connect(self.show_icon_threshold)
        self.ui.reset_text_color_default_button.clicked.connect(self.reset_text_colors)
        self.ui.reset_icon_color_default_button.clicked.connect(self.reset_icon_colors)

        #self.ui.icon_color_button.clicked.connect(self.choose_icon_color)
        #self.ui.text_color_button.clicked.connect(self.choose_text_color)

        self.ui.icon_color_button.clicked.connect(self.matplot_test_icon)
        self.ui.text_color_button.clicked.connect(self.matplot_test_text)

        self.ui.analyze_ee_log_button.clicked.connect(self.analyze_ee_log)
        self.ui.overlay_text_size_slider.valueChanged.connect(self.update_overlay_text_size)
        self.ui.template_match_slider.valueChanged.connect(self.update_template_match_condition)

        self.ui.r_spinner.valueChanged.connect(self.update_rgb_to_hsv_label)
        self.ui.g_spinner.valueChanged.connect(self.update_rgb_to_hsv_label)
        self.ui.b_spinner.valueChanged.connect(self.update_rgb_to_hsv_label)

        if self.ui.overlay_checkbox.isChecked(): self.toggle_overlay()
        if self.ui.sounds_checkbox.isChecked(): self.toggle_sounds()

        self.load_config()
        self.max_time = 120 if self.ui.time_120_radio_button.isChecked() else 156

        # make a list of ui elements to disable when running
        self.ui_elements = [self.ui.start_button,
        self.ui.overlay_checkbox,
        self.ui.sounds_checkbox,
        self.ui.ui_scale_spinner,
        self.ui.enable_affinity_scanner_checkbox,
        self.ui.track_arbitration_drone_checkbox,
        self.ui.icon_color_button,
        self.ui.text_color_button,
        self.ui.reset_icon_color_default_button,
        self.ui.reset_text_color_default_button,
        self.ui.label_5,
        self.ui.ui_scale_spinner,
        self.ui.label_6,
        self.ui.time_120_radio_button,
        self.ui.time_156_radio_button,
        ]

        # border: " + BorderThickness + " solid " +hexColorCode + ";"
        self.ui.smeeta_icon_widget.setStyleSheet( 'QWidget {border: 1 solid hsv(0,0,0); background-color: hsv(%d,%d,%d);}'%(self.icon_color_hsv[0]*2, self.icon_color_hsv[1], self.icon_color_hsv[2]))
        self.ui.text_color_widget.setStyleSheet( 'QWidget {border: 1 solid hsv(0,0,0); background-color: hsv(%d,%d,%d);}'%(self.text_color_hsv[0]*2,self.text_color_hsv[1],self.text_color_hsv[2]))

        self.image_label_list =[self.ui.image_label_1,self.ui.image_label_2,self.ui.image_label_3,self.ui.image_label_4,self.ui.image_label_5]
        self.label_image_label_list =[self.ui.label_image_label_1,self.ui.label_image_label_2,self.ui.label_image_label_3,self.ui.label_image_label_4,self.ui.label_image_label_5]

        # set logging to critical, otherwise matplotlib will put a bunch of garbage
        logging.getLogger('matplotlib.font_manager').disabled = True
        logging.basicConfig(format='[%(asctime)s] %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p', filename='general.log', filemode='w', level=logging.DEBUG)
        logging.info('Program initialized')

        if not os.path.isfile("solNodes.json"):
            os.rename('base_solNodes.json','solNodes.json')

    def load_ui(self):
        loader = QUiLoader()
        path = os.fspath(Path(__file__).resolve().parent / "form.ui")
        ui_file = QFile(path)
        ui_file.open(QFile.ReadOnly)
        self.ui = loader.load(ui_file, self)
        ui_file.close()

    def load_config(self):
        with open('config.json') as json_file:
            data = json.load(json_file)
        self.icon_color_hsv = data['icon_color_hsv']
        self.text_color_hsv = data['text_color_hsv']

    def save_config(self):
        with open('config.json') as json_file:
            data = json.load(json_file)
        data['icon_color_hsv'] = self.icon_color_hsv
        data['text_color_hsv'] = self.text_color_hsv

        with open('config.json', 'w') as outfile:
            json.dump(data, outfile)

    def closeEvent(self, arg):
        self.keep_threads_alive=False
        sys.exit(0)

    def toggle_overlay(self):
        if self.ui.overlay_checkbox.isChecked():
            self.dialog = DisplayDialog()
            self.dialog.show()
        else:
            self.dialog.close()
            self.dialog=None

    def toggle_sounds(self):
        self.sounds = self.ui.sounds_checkbox.isChecked()

    def start_scanning(self):
        if not self.keep_threads_alive:
            self.clear()
            #disable settings buttons
            self.disable_setting_buttons()

            self.keep_threads_alive = True
            if self.ui.enable_affinity_scanner_checkbox.isChecked():
                self.scanner = Scanner(self)
                self.thread_list.append('Scan Thread')
                x1 = threading.Thread(target=self.scan_screen )
                x1.start()

            if self.dialog:
                self.thread_list.append('Dialog Thread')
                x2 = threading.Thread(target=self.update_dialog )
                x2.start()

            if self.sounds:
                self.thread_list.append('Sound Thread')
                x3 = threading.Thread(target=self.flush_sound_queue )
                x3.start()

            if self.ui.track_arbitration_drone_checkbox.isChecked():
                self.thread_list.append('Track Drone Thread')
                x4 = threading.Thread(target=self.scan_ee_logs )
                x4.start()

    def update_dialog(self):
        self.dialog_thread_active = True
        while(self.keep_threads_alive):
            next_expiry = self.scanner.get_next_expiry()
            if next_expiry is not None:
                next_expiry = self.scanner.get_next_expiry()-time.time()
                remaining_chances = int((round(next_expiry,1)+6)/27)
                self.dialog.set_text("\nActive: %d\nNext Expiry: %.1fs\nRemaining Chances: %d"%(self.scanner.get_procs_active(), next_expiry, remaining_chances))
            else:
                self.dialog.set_text("")
            if len(self.scanner.proc_expiry_queue)>0:
                time.sleep((self.scanner.get_next_expiry()-time.time())%1)
            else:
                time.sleep(2)
        print("Dialog thread exit")
        self.dialog_thread_active = False
        self.thread_list.pop()
        if len(self.thread_list)<=0:
            self.enable_setting_buttons()

    def scan_screen(self):
        while(self.keep_threads_alive):
            if self.scanner.template_matching_enabled:
                self.scanner.scan_match_template()
                time.sleep(2)
            else:
                self.scanner.scan_match_text()
                time.sleep(2)
        print("Scan thread exit")
        self.scan_thread_active = False
        self.thread_list.pop()
        if len(self.thread_list)<=0:
            self.enable_setting_buttons()

    def stop_scanning(self):
        self.keep_threads_alive = False

    def flush_sound_queue(self):
        while(self.keep_threads_alive):
            while( len(self.scanner.sound_queue) > 0 and self.keep_threads_alive):
                self.play_s( os.path.join(self.dirname,'Sounds', self.scanner.sound_queue[0] ) , bl=True)
                self.scanner.sound_queue.popleft()
            time.sleep(5)
        print("Sound thread exit")
        self.sound_thread_active = False
        self.thread_list.pop()
        if len(self.thread_list)<=0:
            self.enable_setting_buttons()

    def play_s(self, file_name, bl = False):
        try:
            playsound(file_name, bl)
        except Exception as e:
            print(e)
            print("Sound not found! -> ",file_name)

    def disable_setting_buttons(self):
        for elem in self.ui_elements:
            elem.setEnabled(False)

    def enable_setting_buttons(self):
        for elem in self.ui_elements:
            elem.setEnabled(True)

    def choose_text_color(self):
        self.color_event = TEXT_COLOR_TYPE
        self.wait_for_color_selection()
        #self.wait_for_color_selection_matplotlib()
    def choose_icon_color(self):
        self.color_event = SMEETA_ICON_COLOR_TYPE
        self.wait_for_color_selection()

    def matplot_test_icon(self):
        self.color_event = SMEETA_ICON_COLOR_TYPE
        self.wait_for_color_selection_matplotlib()
    def matplot_test_text(self):
        self.color_event = TEXT_COLOR_TYPE
        self.wait_for_color_selection_matplotlib()

    def wait_for_color_selection(self):
        self.screenshot = self.screencap.get_screenshot()
        cv2.namedWindow('Select Color', cv2.WINDOW_NORMAL)
        cv2.setWindowProperty('Select Color', cv2.WND_PROP_FULLSCREEN, 1)
        cv2.setMouseCallback('Select Color', self.color_select_mouse_event)

        cv2.imshow('Select Color',self.screenshot)

        while(1):
            #cv2.imshow('Select Color',self.screenshot)
            # press escape to exit
            if cv2.waitKey(20) & 0xFF == 27:
                break
        cv2.destroyAllWindows()
    def color_select_mouse_event(self, event, x, y, flags, params):
        if event == cv2.EVENT_LBUTTONDOWN:
            bgr_pixel = self.screenshot[y,x]
            # convert pixel to hsv
            hsv_pixel = cv2.cvtColor(self.screenshot, cv2.COLOR_BGR2HSV)[y,x]
            print("HSV: %s, BGR: %s"%(str(hsv_pixel),str(bgr_pixel)))
            # set widget to color

            cv2.destroyAllWindows()
            if self.color_event == TEXT_COLOR_TYPE:
                self.text_color_hsv = hsv_pixel.tolist()
                self.ui.text_color_widget.setStyleSheet( 'QWidget {border: 1 solid hsv(0,0,0); background-color: hsv(%d,%d,%d);}'%(hsv_pixel[0]*2, hsv_pixel[1], hsv_pixel[2]))
            else:
                self.icon_color_hsv = hsv_pixel.tolist()
                self.ui.smeeta_icon_widget.setStyleSheet( 'QWidget {border: 1 solid hsv(0,0,0); background-color: hsv(%d,%d,%d);}'%(hsv_pixel[0]*2, hsv_pixel[1], hsv_pixel[2]))
            self.save_config()

    def wait_for_color_selection_matplotlib(self):
        self.screenshot = self.screencap.get_screenshot()
        self.screenshot = cv2.cvtColor(self.screenshot, cv2.COLOR_BGR2RGB)
        self.screenshot_hsv = cv2.cvtColor(self.screenshot, cv2.COLOR_RGB2HSV)

        fig, ax = plt.subplots()
        #fig.canvas.manager.full_screen_toggle()
        wm = plt.get_current_fig_manager()
        #wm.window.state('zoomed')
        ax.imshow(self.screenshot)

        cid = fig.canvas.mpl_connect('button_press_event', self.onclick)
        plt.show()
    def onclick(self, event):
        if event.button == 3:
            hsv_pixel = self.screenshot_hsv[int(round(event.ydata,0)), int(round(event.xdata,0))]
            print('%s click: button=%d, x=%d, y=%d, HSV:[%d, %d, %d]' %('double' if event.dblclick else 'single', event.button,event.xdata, event.ydata, hsv_pixel[0], hsv_pixel[1], hsv_pixel[2]))

            if self.color_event == TEXT_COLOR_TYPE:
                self.text_color_hsv = hsv_pixel.tolist()
                self.ui.text_color_widget.setStyleSheet( 'QWidget {border: 1 solid hsv(0,0,0); background-color: hsv(%d,%d,%d);}'%(hsv_pixel[0]*2, hsv_pixel[1], hsv_pixel[2]))
            else:
                self.icon_color_hsv = hsv_pixel.tolist()
                self.ui.smeeta_icon_widget.setStyleSheet( 'QWidget {border: 1 solid hsv(0,0,0); background-color: hsv(%d,%d,%d);}'%(hsv_pixel[0]*2, hsv_pixel[1], hsv_pixel[2]))
            self.save_config()
            plt.close()

    def show_icon_threshold(self):
        self.scanner = Scanner(self)
        self.scanner.show_icon_threshold()
    def show_text_threshold(self):
        self.scanner = Scanner(self)
        self.scanner.show_text_threshold()

    def reset_text_colors(self):
        self.text_color_hsv = [0, 0, 255]
        self.ui.text_color_widget.setStyleSheet( 'QWidget {border: 1 solid hsv(0,0,0); background-color: hsv(%d,%d,%d);}'%(0,0,255))
        self.save_config()
    def reset_icon_colors(self):
        self.icon_color_hsv = [95, 255, 255]
        self.ui.smeeta_icon_widget.setStyleSheet( 'QWidget {border: 1 solid hsv(0,0,0); background-color: hsv(%d,%d,%d);}'%(95*2,255,255))
        self.save_config()

    def scan_ee_logs(self):
        eep = EeLogParser(self.max_time, self.ui)
        while(self.keep_threads_alive):
            try:
                eep.parse_file()
            except Exception as e:
                print("Falied to read ee log: %s"%(str(e)))
            self.ui.drone_spawns_label.setText(str(eep.drone_spawns))
            self.ui.total_spawns_label.setText(str(eep.total_spawns))
            self.ui.mission_time_label.setText(str(datetime.timedelta(seconds=int(eep.mission_time))))
            if eep.mission_time>0:
                self.ui.drone_kpm_label.setText('%.2f, %.2f'%(eep.drone_spawns/(eep.mission_time/60), eep.drone_spawns/(eep.mission_time/3600)))
                self.ui.kpm_label.setText('%.2f'%(eep.total_spawns/(eep.mission_time/60)))
            if eep.total_spawns>0:
                self.ui.drone_rate_label.setText('%.2f%%'%(100*eep.drone_spawns/(eep.total_spawns-eep.drone_spawns)))
            else: self.ui.drone_rate_label.setText('-')
            # update dialog
            if self.dialog is not None:
                if eep.in_mission and eep.drone_spawns>0:
                    time_formatted = time.strftime('%H:%M:%S', time.localtime(eep.latest_log_time+eep.global_time))
                    #eep.status_text='Drone spawned %d seconds ago\n%d drones total\nLogs updated %s (%d seconds ago)'%(time.time()-eep.last_spawn_time,eep.drone_spawns, time_formatted, time.time()-(eep.latest_log_time+eep.global_time))
                    disp_str=""
                    if self.ui.dt1_checkbox.isChecked(): disp_str+="\nTotal Drones: %d"%eep.drone_spawns
                    if self.ui.dt2_checkbox.isChecked(): disp_str+="\nDrones Per Hour: %d"%(eep.drone_spawns/((eep.latest_log_time-(eep.mission_start_time-eep.global_time))/3600))
                    if self.ui.dt3_checkbox.isChecked(): disp_str+="\nDrone spawned %d seconds ago"%(time.time()-eep.last_spawn_time,eep.drone_spawns)
                    if self.ui.dt4_checkbox.isChecked(): disp_str+="\nLogs updated %s (%d seconds ago)"%(time_formatted, time.time()-(eep.latest_log_time+eep.global_time))

                    if self.ui.dt5_checkbox.isChecked(): disp_str+="\n\nCurrent Arbitration: %s"%eep.current_arbitration

                    eep.status_text=disp_str
                else:
                    #print(ee)
                    eep.status_text="Current Arbitration: %s"%eep.current_arbitration
                self.dialog.set_arb_text(eep.status_text)
            time.sleep(1)
        print('EE log parse thread exit')
        self.thread_list.pop()
        if len(self.thread_list)<=0:
            self.enable_setting_buttons()

    def analyze_ee_log(self):
        eep = EeLogParser(self.max_time, self.ui)
        eep.plot_logs()

    def update_overlay_text_size(self):
        if self.dialog:
            self.dialog.set_text_size(13*(1+self.ui.overlay_text_size_slider.value()/100))

    def update_template_match_condition(self):
        if self.scanner:
            self.scanner.template_match_threshold = self.ui.template_match_slider.value()*0.6/100
            self.scanner.template_matching_enabled = False if self.scanner.template_match_threshold == 0 else True
        self.ui.template_match_label.setText("%d%%"%(self.ui.template_match_slider.value()*0.6))

    def update_rgb_to_hsv_label(self):
        r = self.ui.r_spinner.value()/255
        g = self.ui.g_spinner.value()/255
        b = self.ui.b_spinner.value()/255
        cmax=  max(r,g,b)
        #print(cmax)
        cmin=min(r,g,b)
        delta=cmax-cmin
        if delta ==0:
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
        self.ui.hsv_label.setText('(%d,%d,%d)'%(h,s*255,v*255))

class DisplayDialog(QtWidgets.QDialog):
    def __init__(self):
        super(DisplayDialog, self).__init__()

        self.setAttribute(QtCore.Qt.WA_TransparentForMouseEvents)
        self.setAttribute(QtCore.Qt.WA_TranslucentBackground, True)
        #self.setAttribute(QtCore.Qt.WA_AlwaysStackOnTop, True)
        self.setWindowFlags(QtCore.Qt.FramelessWindowHint | QtCore.Qt.WindowStaysOnTopHint)
        #self.setWindowFlags()

        layout = QtWidgets.QVBoxLayout(self)
        self.label = QtWidgets.QLabel('')
        self.label.setFont(QtGui.QFont('Arial', 13))
        self.label.setStyleSheet("color : white")
        layout.addWidget(self.label)

        self.arblabel = QtWidgets.QLabel('')
        self.arblabel.setFont(QtGui.QFont('Arial', 13))
        self.arblabel.setStyleSheet("color : white")
        layout.addWidget(self.arblabel)
        self.move(0, 1080//2)

        self.label_list=[self.arblabel, self.label]

    def set_text(self, text):
        if GetWindowText(GetForegroundWindow()) == 'Warframe':
            self.label.setText(text)
        else:
            self.label.setText('')
    def set_text_size(self, size):
        for label in self.label_list:
            self.arblabel.setFont(QtGui.QFont('Arial', int(size)))
    def set_arb_text(self, text):
        if GetWindowText(GetForegroundWindow()) == 'Warframe':
            self.arblabel.setText(text)
        else:
            self.arblabel.setText('')

if __name__ == "__main__":
    app = QApplication([])
    widget = MainWindow()
    widget.show()
    sys.exit(app.exec())



