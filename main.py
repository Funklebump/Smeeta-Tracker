import os
from pathlib import Path
import sys
import time
import json
import datetime

from win32gui import GetWindowText, GetForegroundWindow

from PySide6.QtWidgets import QApplication, QWidget
from PySide6 import QtWidgets
from PySide6 import QtGui
from PySide6 import QtCore
from PySide6.QtCore import QFile
from PySide6.QtUiTools import QUiLoader
from monitor import Monitor

from windowcapture import WindowCapture
import cv2
from matplotlib import pyplot as plt
import ctypes
import requests
import subprocess
import inspect
from PySide6.QtWidgets import QComboBox, QCheckBox, QRadioButton, QSpinBox, QSlider
 
SMEETA_ICON_COLOR_TYPE = 'SMEETA_ICON_COLOR_TYPE'
TEXT_COLOR_TYPE = 'TEXT_COLOR_TYPE'
version_link = "https://raw.github.com/A-DYB/smeeta-tracker-2/main/version.json"

class MainWindow(QWidget):
    def __init__(self):
        self.script_folder = Path(__file__).parent.absolute()

        super(MainWindow, self).__init__()
        self.ui=None
        self.load_ui()

        self.affinity_proc_duration = 120 if self.ui.time_120_radio_button.isChecked() else 156
        self.duration_scale = 1 if self.ui.time_120_radio_button.isChecked() else 1.3
        self.ui.time_120_radio_button.toggled.connect(self.update_settings)
        self.drop_chance_booster = 2 if self.ui.drop_chance_booster_checkbox.isChecked() else 1
        self.ui.drop_chance_booster_checkbox.stateChanged.connect(self.update_settings)
        self.drop_booster = 2 if self.ui.drop_booster_checkbox.isChecked() else 1
        self.ui.drop_booster_checkbox.stateChanged.connect(self.update_settings)
        self.drop_booster2 = 2 if self.ui.drop_booster_checkbox_2.isChecked() else 1
        self.ui.drop_booster_checkbox_2.stateChanged.connect(self.update_settings)
        self.bless_booster = 1.25 if self.ui.bless_booster_checkbox.isChecked() else 1
        self.ui.bless_booster_checkbox.stateChanged.connect(self.update_settings)
        self.dark_sector_booster = self.ui.dark_sector_booster_spinner.value()
        self.ui.dark_sector_booster_spinner.valueChanged.connect(self.update_settings)
        self.ui_scale = float(self.ui.ui_scale_combo.currentText())
        self.ui.ui_scale_combo.currentIndexChanged.connect(self.update_settings)
        self.template_match_threshold = self.ui.template_match_slider.value()/100
        self.ui.template_match_slider.sliderReleased.connect(self.update_settings)
        self.scanner_refresh_rate_s = 0.5+2*self.ui.refresh_rate_slider.value()/100

        self.warframe_window_found=False

        self.image_label_list = [self.ui.image_label_1,self.ui.image_label_2,self.ui.image_label_3,self.ui.image_label_4,self.ui.image_label_5]
        self.label_image_label_list = [self.ui.label_image_label_1,self.ui.label_image_label_2,self.ui.label_image_label_3,self.ui.label_image_label_4,self.ui.label_image_label_5]

        self.image_list = []
        self.image_text_list = []
        self.image_index=0
        self.max_entities = 5

        user32 = ctypes.windll.user32
        self.screen_capture = WindowCapture('Warframe', ( user32.GetSystemMetrics(0) , user32.GetSystemMetrics(1) ), self.ui )

        self.icon_color_hsv = [95, 255, 255]
        self.text_color_hsv = [0, 0, 255]

        self.overlay = Overlay()
        self.overlay.show()
        self.monitor = Monitor(self)

        self.check_for_updates()

        app.aboutToQuit.connect(self.closeEvent)

        self.ui.update_button.clicked.connect(self.spawn_updater_and_die)

        self.ui.start_button.clicked.connect(self.monitor.start_scanning)
        self.ui.stop_button.clicked.connect(self.monitor.stop_scanning)

        self.ui.reset_text_color_default_button.clicked.connect(self.reset_text_colors)
        self.ui.reset_icon_color_default_button.clicked.connect(self.reset_icon_colors)
        self.ui.ui_scale_combo.currentIndexChanged.connect(self.save_config)

        self.ui.icon_color_button.clicked.connect(lambda : self.select_color(SMEETA_ICON_COLOR_TYPE))
        self.ui.text_color_button.clicked.connect(lambda : self.select_color(TEXT_COLOR_TYPE))

        self.ui.analyze_ee_log_button.clicked.connect(self.monitor.log_parser.plot_logs)
        self.ui.overlay_text_size_slider.valueChanged.connect(lambda : self.overlay.set_text_size(13*(1+self.ui.overlay_text_size_slider.value()/100)))
        self.ui.template_match_slider.valueChanged.connect(self.update_template_match_condition)
        self.ui.refresh_rate_slider.valueChanged.connect(self.update_scan_refresh_rate)


        self.ui.r_spinner.valueChanged.connect(self.update_rgb_to_hsv_label)
        self.ui.g_spinner.valueChanged.connect(self.update_rgb_to_hsv_label)
        self.ui.b_spinner.valueChanged.connect(self.update_rgb_to_hsv_label)

        self.ui.test_bounds_button.clicked.connect(self.display_detection_area)
        #self.ui.test_bounds_button.clicked.connect(self.test_template_match)
        self.ui.test_icon_threshold_button.clicked.connect(self.display_icon_filter)
        self.ui.test_text_threshold_button.clicked.connect(self.display_text_filter)

        self.load_config()

        # make a list of ui elements to disable when running
        self.ui_elements = [    self.ui.start_button,
                                self.ui.overlay_checkbox,
                                self.ui.sounds_checkbox,
                                self.ui.ui_scale_combo,
                                self.ui.enable_affinity_scanner_checkbox,
                                self.ui.icon_color_button,
                                self.ui.text_color_button,
                                self.ui.reset_icon_color_default_button,
                                self.ui.reset_text_color_default_button,
                                self.ui.label_5,
                                self.ui.label_6,
                                self.ui.time_120_radio_button,
                                self.ui.time_156_radio_button,
                                self.ui.test_bounds_button,
                            ]

        # border: " + BorderThickness + " solid " +hexColorCode + ";"
        self.ui.smeeta_icon_widget.setStyleSheet( 'QWidget {border: 1 solid hsv(0,0,0); background-color: hsv(%d,%d,%d);}'%(self.icon_color_hsv[0]*2, self.icon_color_hsv[1], self.icon_color_hsv[2]))
        self.ui.text_color_widget.setStyleSheet( 'QWidget {border: 1 solid hsv(0,0,0); background-color: hsv(%d,%d,%d);}'%(self.text_color_hsv[0]*2,self.text_color_hsv[1],self.text_color_hsv[2]))

        if not os.path.isfile(os.path.join(self.script_folder,"solNodes.json")):
            os.rename(os.path.join(self.script_folder,'base_solNodes.json'),os.path.join(self.script_folder,'solNodes.json'))

        self.guirestore(QtCore.QSettings(os.path.join(self.script_folder, 'saved_settings.ini'), QtCore.QSettings.IniFormat))

        self.update_template_match_condition()
        self.update_scan_refresh_rate()
        os.system('cls')

    def update_settings(self):
        self.affinity_proc_duration = 120 if self.ui.time_120_radio_button.isChecked() else 156
        self.duration_scale = 1 if self.ui.time_120_radio_button.isChecked() else 1.3
        self.drop_chance_booster = 2 if self.ui.drop_chance_booster_checkbox.isChecked() else 1
        self.drop_booster = 2 if self.ui.drop_booster_checkbox.isChecked() else 1
        self.drop_booster2 = 2 if self.ui.drop_booster_checkbox_2.isChecked() else 1
        self.bless_booster = 1.25 if self.ui.bless_booster_checkbox.isChecked() else 1
        self.dark_sector_booster = self.ui.dark_sector_booster_spinner.value()

        self.ui_scale = float(self.ui.ui_scale_combo.currentText())
        self.template_match_threshold = self.ui.template_match_slider.value()/100
        self.scanner_refresh_rate_s = 0.5+2*self.ui.refresh_rate_slider.value()/100

        self.monitor.screen_scanner.update_ui_settings()

    def check_for_updates(self):
        self.latest_version_json = json.loads(requests.get(version_link).text)
        with open(os.path.join(self.script_folder,"version.json")) as f:
            version_json = json.load(f)
        if self.latest_version_json["version"] != version_json["version"]:
            self.ui.version_label.setText('''<a href='https://github.com/A-DYB/smeeta-tracker-2/releases'>Updates available!</a>''')
            self.ui.version_label.setOpenExternalLinks(True)
            self.ui.update_button.setEnabled(True)
            self.ui.update_button.show()
        else:
            self.ui.version_label.setText("Version:%s - No updates available"%version_json["version"])
            self.ui.update_button.setEnabled(False)
            self.ui.update_button.hide()

    def spawn_updater_and_die(self, exit_code=0):
        """
        Start an external program and exit the script
        with the specified return code.

        Takes the parameter program, which is a list
        that corresponds to the argv of your command.
        """
        self.monitor.monitor_game = False
        while len(self.monitor.thread_list)>0:
            time.sleep(1)
        # Start the external program
        subprocess.Popen(['python', './updater.py'])
        # We have started the program, and can suspend this interpreter
        sys.exit(exit_code)

    def load_ui(self):
        loader = QUiLoader()
        path = os.fspath(Path(__file__).resolve().parent / "form.ui")
        ui_file = QFile(path)
        ui_file.open(QFile.ReadOnly)
        self.ui = loader.load(ui_file, self)
        ui_file.close()

    def load_config(self):
        self.create_config()
        with open(os.path.join(self.script_folder,'config.json')) as json_file:
            data = json.load(json_file)
        self.icon_color_hsv = data['icon_color_hsv']
        self.text_color_hsv = data['text_color_hsv']

        index = self.ui.ui_scale_combo.findText('%.1f'%(data['in_game_hud_scale']), QtCore.Qt.MatchFixedString)
        if index >= 0:
            self.ui.ui_scale_combo.setCurrentIndex(index)

    def save_config(self):
        self.create_config()

        with open(os.path.join(self.script_folder,'config.json')) as json_file:
            data = json.load(json_file)
        data['icon_color_hsv'] = self.icon_color_hsv
        data['text_color_hsv'] = self.text_color_hsv
        if is_float(self.ui.ui_scale_combo.currentText()):
            data["in_game_hud_scale"] = float(self.ui.ui_scale_combo.currentText())
        else:
            data["in_game_hud_scale"] = 1

        with open(os.path.join(self.script_folder,'config.json'), 'w') as outfile:
            json.dump(data, outfile)
    
    def create_config(self):
        if not os.path.isfile(os.path.join(self.script_folder,'config.json')):
            data = {"icon_color_hsv": [95, 255, 255], "text_color_hsv": [0, 0, 255], "in_game_hud_scale": 1}
            with open('config.json', 'w') as f:
                json.dump(data, f)

    def closeEvent(self, arg):
        self.guisave(QtCore.QSettings(os.path.join(self.script_folder, 'saved_settings.ini'), QtCore.QSettings.IniFormat))
        self.monitor.monitor_game = False
        sys.exit(0)

    def disable_setting_buttons(self):
        for elem in self.ui_elements:
            elem.setEnabled(False)

    def enable_setting_buttons(self):
        for elem in self.ui_elements:
            elem.setEnabled(True)

    def select_color(self, selection_type):
        screenshot_bgr = self.screen_capture.get_screenshot()
        screenshot_rgb = cv2.cvtColor(screenshot_bgr, cv2.COLOR_BGR2RGB)
        screenshot_hsv = cv2.cvtColor(screenshot_bgr, cv2.COLOR_BGR2HSV)

        fig, ax = plt.subplots()
        wm = plt.get_current_fig_manager()
        ax.imshow(screenshot_rgb)
        cid = fig.canvas.mpl_connect('button_press_event', lambda event: self.onclick(event, selection_type, screenshot_hsv))
        plt.suptitle(f'Right click to select {selection_type}.\nColor must be selected from:\nWarframe settings -> Accessibility -> Customize UI colors')
        plt.tight_layout()
        plt.show()

    def onclick(self, event, selection_type, screenshot_hsv):
        if event.button == 3:
            hsv_pixel = screenshot_hsv[int(round(event.ydata,0)), int(round(event.xdata,0))]
            print('%s click: button=%d, x=%d, y=%d, HSV:[%d, %d, %d]' %('double' if event.dblclick else 'single', event.button,event.xdata, event.ydata, hsv_pixel[0], hsv_pixel[1], hsv_pixel[2]))

            if selection_type == TEXT_COLOR_TYPE:
                self.text_color_hsv = hsv_pixel.tolist()
                self.ui.text_color_widget.setStyleSheet( 'QWidget {border: 1 solid hsv(0,0,0); background-color: hsv(%d,%d,%d);}'%(hsv_pixel[0]*2, hsv_pixel[1], hsv_pixel[2]))
            else:
                self.icon_color_hsv = hsv_pixel.tolist()
                self.ui.smeeta_icon_widget.setStyleSheet( 'QWidget {border: 1 solid hsv(0,0,0); background-color: hsv(%d,%d,%d);}'%(hsv_pixel[0]*2, hsv_pixel[1], hsv_pixel[2]))
            self.save_config()
            plt.close()
            self.monitor.screen_scanner.update_ui_settings()

    def reset_text_colors(self):
        self.text_color_hsv = [0, 0, 255]
        self.ui.text_color_widget.setStyleSheet( 'QWidget {border: 1 solid hsv(0,0,0); background-color: hsv(%d,%d,%d);}'%(0,0,255))
        self.save_config()

    def reset_icon_colors(self):
        self.icon_color_hsv = [95, 255, 255]
        self.ui.smeeta_icon_widget.setStyleSheet( 'QWidget {border: 1 solid hsv(0,0,0); background-color: hsv(%d,%d,%d);}'%(95*2,255,255))
        self.save_config()

    def update_template_match_condition(self):
        self.monitor.screen_scanner.template_match_threshold = self.ui.template_match_slider.value()/100
        self.template_match_threshold = self.ui.template_match_slider.value()/100
        self.ui.template_match_label.setText("%d%%"%(self.ui.template_match_slider.value()))

    def update_scan_refresh_rate(self):
        self.monitor.screen_scanner.refresh_rate_s = 0.5+2*self.ui.refresh_rate_slider.value()/100
        self.scanner_refresh_rate_s = 0.5+2*self.ui.refresh_rate_slider.value()/100
        self.ui.refresh_rate_label.setText(f'{self.monitor.screen_scanner.refresh_rate_s:.1f}s')

    def update_rgb_to_hsv_label(self):
        r = self.ui.r_spinner.value()/255
        g = self.ui.g_spinner.value()/255
        b = self.ui.b_spinner.value()/255
        cmax=max(r,g,b)
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

    def guisave(self, settings):
        # Save geometry
        settings.setValue('size', self.size())
        settings.setValue('pos', self.pos())

        for name, obj in inspect.getmembers(self.ui):
            # if type(obj) is QComboBox:  # this works similar to isinstance, but missed some field... not sure why?
            if isinstance(obj, QComboBox):
                name = obj.objectName()  # get combobox name
                index = obj.currentIndex()  # get current index from combobox
                text = obj.itemText(index)  # get the text for current index
                settings.setValue(name, text)  # save combobox selection to registry

            if isinstance(obj, QCheckBox):
                name = obj.objectName()
                state = obj.isChecked()
                settings.setValue(name, state)

            if isinstance(obj, QRadioButton):
                name = obj.objectName()
                value = obj.isChecked()  # get stored value from registry
                settings.setValue(name, value)
            if isinstance(obj, QSpinBox):
                name  = obj.objectName()
                value = obj.value()             # get stored value from registry
                settings.setValue(name, value)

            if isinstance(obj, QSlider):
                name  = obj.objectName()
                value = obj.value()             # get stored value from registry
                settings.setValue(name, value)


    def guirestore(self, settings):
        # Restore geometry  
        #self.resize(settings.value('size', QtCore.QSize(500, 500)))
        self.move(settings.value('pos', QtCore.QPoint(60, 60)))

        for name, obj in inspect.getmembers(self.ui):
            if isinstance(obj, QComboBox):
                index = obj.currentIndex()  # get current region from combobox
                # text   = obj.itemText(index)   # get the text for new selected index
                name = obj.objectName()

                value = (settings.value(name))

                if value == "":
                    continue

                index = obj.findText(value)  # get the corresponding index for specified string in combobox

                if index == -1:  # add to list if not found
                    obj.insertItems(0, [value])
                    index = obj.findText(value)
                    obj.setCurrentIndex(index)
                else:
                    obj.setCurrentIndex(index)  # preselect a combobox value by index
                    
            if isinstance(obj, QCheckBox):
                name = obj.objectName()
                value = settings.value(name)  # get stored value from registry
                if value != None:
                    obj.setChecked(str2bool(value))  # restore checkbox

            if isinstance(obj, QRadioButton):
                name = obj.objectName()
                value = settings.value(name)  # get stored value from registry
                if value != None:
                    obj.setChecked(str2bool(value))
            
            if isinstance(obj, QSlider):
                name = obj.objectName()
                value = settings.value(name)    # get stored value from registry
                if value != None:           
                    obj. setValue(int(value))   # restore value from registry

            if isinstance(obj, QSpinBox):
                name = obj.objectName()
                value = settings.value(name)    # get stored value from registry
                if value != None:
                    obj. setValue(int(value))   # restore value from registry
    
    def on_data_ready(self):
        self.overlay.scan_label_group.reset_text()
        self.overlay.log_label_group.reset_text()

        for i in range(len(self.image_label_list)):
            if i < len(self.image_list):
                self.image_label_list[i].setPixmap(self.image_list[i])
                self.label_image_label_list[i].setText(self.image_text_list[i])
            else:
                self.image_label_list[i].clear()
                self.label_image_label_list[i].setText("")
        #self.image_list = []
        #self.image_text_list = []

        if self.monitor.screen_scanner.paused:
            self.overlay.scan_label_group.add_text("Paused")
        else:
            next_affinity_expiry_unix = None
            next_affinity_expiry_unix = self.monitor.screen_scanner.get_next_expiry_unix()
            if next_affinity_expiry_unix:
                next_affinity_expiry_s = next_affinity_expiry_unix - time.time()
                active_procs = len(self.monitor.screen_scanner.affinity_proc_list)
                remaining_chances = int((round(next_affinity_expiry_s,1)+6)/27)

                self.overlay.scan_label_group.add_text(f'Active: {active_procs}')
                self.overlay.scan_label_group.add_text(f'Next Expiry: {int(next_affinity_expiry_s)}')
                self.overlay.scan_label_group.add_text(f'Remaining Chances: {remaining_chances}')
                
                # update main window labels
                self.ui.active_label.setText(f'{active_procs}')
                self.ui.total_boost_label.setText(f'{2**active_procs}')
                self.ui.next_expiry_label.setText(f'{next_affinity_expiry_s}')
                self.ui.extra_proc_chances_label.setText(f'{remaining_chances}')
            else:
                self.ui.active_label.setText(f'{0}')
                self.ui.total_boost_label.setText(f'{1}')
                self.ui.next_expiry_label.setText(f'-')
                self.ui.extra_proc_chances_label.setText(f'-')

            if self.monitor.log_parser.in_mission and self.ui.charm_rotation_checkbox.isChecked(): 
                ref_timestamp = max(self.monitor.log_parser.mission_start_timestamp_unix_s+1, self.monitor.screen_scanner.proc_validator.last_proc_reference_timestamp_unix_s)
                self.overlay.scan_label_group.add_text(f'Charm Rotation: {(27.4-(time.time() - (ref_timestamp))%27.4):.1f}s')

            # update ui labels
            self.ui.drone_spawns_label.setText(str(self.monitor.log_parser.drone_spawns))
            self.ui.total_spawns_label.setText(str(self.monitor.log_parser.enemy_spawns))
            self.ui.mission_time_label.setText(str(datetime.timedelta(seconds=int(self.monitor.log_parser.mission_duration_s))))

            if self.monitor.log_parser.mission_duration_s>0:
                self.ui.drone_kpm_label.setText('%.2f, %.2f'%(self.monitor.log_parser.drone_spawns/(self.monitor.log_parser.mission_duration_s/60), self.monitor.log_parser.drone_spawns/(self.monitor.log_parser.mission_duration_s/3600)))
                self.ui.kpm_label.setText('%.2f'%(self.monitor.log_parser.enemy_spawns/(self.monitor.log_parser.mission_duration_s/60)))

            if self.monitor.log_parser.enemy_spawns>0:
                self.ui.drone_rate_label.setText('%.2f%%'%(100*self.monitor.log_parser.drone_spawns/self.monitor.log_parser.enemy_spawns))
            else: 
                self.ui.drone_rate_label.setText('-')

            # Update drone count in overlay
            if self.monitor.log_parser.in_mission and self.monitor.log_parser.drone_spawns>0:
                if self.ui.dt1_checkbox.isChecked(): 
                    self.overlay.log_label_group.add_text(f'Drones: {self.monitor.log_parser.drone_spawns} ({int(self.monitor.log_parser.drones_per_hour)}/hr)')

            # Display current arbitration in overlay
            if self.ui.dt5_checkbox.isChecked(): 
                mission_info_str = self.monitor.log_parser.get_node_info_string(self.monitor.log_parser.current_arbitration)
                self.overlay.log_label_group.add_text(f'Current Arbitration: {mission_info_str} ({self.monitor.log_parser.current_arbitration})')

            if self.ui.display_tmatch_checkbox.isChecked():
                self.overlay.scan_label_group.add_text(self.monitor.screen_scanner.template_match_status_text)

        if self.warframe_window_found:
            self.ui.window_capture_status_label.setText("Warframe window found!")
        else:
            self.ui.window_capture_status_label.setText("Warframe window not found...")

    def add_detection_info(self, image, image_text):
        if len(self.image_list) < self.max_entities:
            self.image_list.append(image)
            self.image_text_list.append(image_text)
            self.image_index += 1
            if self.image_index > self.max_entities-1:
                self.image_index=0
        else:
            #print(f'Replacing image {self.image_index}')
            self.image_list[self.image_index] = image
            self.image_text_list[self.image_index] = image_text
            self.image_index += 1
            if self.image_index > self.max_entities-1:
                self.image_index=0

    def test_template_match(self):
        screenshot = self.monitor.screen_scanner.wincap.get_screenshot()
        self.monitor.screen_scanner.template_match(screenshot, plot=True)
    
    def display_detection_area(self):
        self.monitor.screen_scanner.display_detection_area()

    def display_icon_filter(self):
        self.monitor.screen_scanner.display_icon_filter()

    def display_text_filter(self):
        self.monitor.screen_scanner.display_text_filter()


class Overlay(QtWidgets.QDialog):
    def __init__(self):
        super(Overlay, self).__init__()

        self.setAttribute(QtCore.Qt.WA_TransparentForMouseEvents)
        self.setAttribute(QtCore.Qt.WA_TranslucentBackground, True)
        self.setWindowFlags(QtCore.Qt.FramelessWindowHint | QtCore.Qt.WindowStaysOnTopHint)

        layout = QtWidgets.QVBoxLayout(self)
        self.scan_label_group = self.LabelGroup(layout)
        self.log_label_group = self.LabelGroup(layout)
  
        self.move(0, 1080//2)

    def set_text_size(self, size):
        self.scan_label_group.set_text_size(size)
        self.log_label_group.set_text_size(size)

    class LabelGroup():
        def __init__(self, layout) -> None:
            self.label_list = []
            self.occupied_labels = 0
            self.count=0
            for i in range(5):
                self.label_list.append( QtWidgets.QLabel(''))
                self.label_list[i].setFont(QtGui.QFont('Arial', 13))
                self.label_list[i].setStyleSheet("color : white")
                layout.addWidget(self.label_list[i])

        def add_text(self, text):
            if GetWindowText(GetForegroundWindow()) == 'Warframe':
                if self.occupied_labels >= len(self.label_list):
                    self.reset_text()
                self.label_list[self.occupied_labels].setText(text)
                self.occupied_labels += 1

        def reset_text(self):
            for lbl in self.label_list:
                #lbl.setText('.'*self.count)
                lbl.setText('')
            self.count +=1
            if self.count >5:
                self.count=0
            self.occupied_labels = 0

        def set_text_size(self, size):
            for lbl in self.label_list:
                lbl.setFont(QtGui.QFont('Arial', int(size)))

def is_float(element: any) -> bool:
    #If you expect None to be passed:
    if element is None: 
        return False
    try:
        float(element)
        return True
    except ValueError:
        return False

def str2bool(v):
  return v.lower() in ("yes", "true", "t", "1")

def hsv_to_rgb(h, s, v):
    if s == 0.0: return (v, v, v)
    i = int(h*6.) # XXX assume int() truncates!
    f = (h*6.)-i; p,q,t = v*(1.-s), v*(1.-s*f), v*(1.-s*(1.-f)); i%=6
    if i == 0: return (v, t, p)
    if i == 1: return (q, v, p)
    if i == 2: return (p, v, t)
    if i == 3: return (p, q, v)
    if i == 4: return (t, p, v)
    if i == 5: return (v, p, q)

if __name__ == "__main__":
    QtCore.QCoreApplication.setAttribute(QtCore.Qt.AA_ShareOpenGLContexts)
    app = QApplication([])
    widget = MainWindow()
    widget.show()

    sys._excepthook = sys.excepthook 
    def exception_hook(exctype, value, traceback):
        print(exctype, value, traceback)
        sys._excepthook(exctype, value, traceback) 
        sys.exit(1) 
    sys.excepthook = exception_hook 

    sys.exit(app.exec())



