import os
from pathlib import Path
import sys
import time
import json
import datetime
from win32gui import GetWindowText, GetForegroundWindow

from PySide6.QtWidgets import QApplication, QWidget, QDoubleSpinBox, QFrame, QMessageBox
from PySide6 import QtWidgets, QtGui, QtCore
from PySide6.QtCore import Signal, QFile
from PySide6.QtGui import QMouseEvent
from PySide6.QtUiTools import QUiLoader
from monitor import Monitor

from windowcapture import WindowCapture
import cv2
import ctypes
import requests
import inspect
from PySide6.QtWidgets import QComboBox, QCheckBox, QRadioButton, QSpinBox, QSlider
from pynput import keyboard
import pickle
import pandas as pd
import logging
import numpy as np
import scanner
import matplotlib
matplotlib.use('Qt5Agg')
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.figure import Figure
import seaborn as sns
from scanner import ProcValidator

version_link = "https://raw.github.com/A-DYB/smeeta-tracker-2/main/version.json"

logger = logging.getLogger('smeeta')
logger.setLevel(logging.DEBUG)
# create file handler which logs even debug messages
fh = logging.FileHandler('exceptions.log')
fh.setLevel(logging.DEBUG)
fh.formatter = logging.Formatter(fmt='%(levelname)s %(asctime)s - %(message)s', datefmt='%d-%b-%y %H:%M:%S')
logger.addHandler(fh)

class MainWindow(QWidget):
    def __init__(self):
        super(MainWindow, self).__init__()
        self.load_ui()
        app.aboutToQuit.connect(self.closeEvent)
        self.setWindowIcon(QtGui.QIcon(':/icons/charm.png'))
        self.setWindowTitle("Smeeta Tracker")

        sys._excepthook = sys.excepthook 
        def exception_hook(exctype, value, traceback):
            logger.exception(value)
            sys._excepthook(exctype, value, traceback) 
            sys.exit(1) 
        sys.excepthook = exception_hook 

        self.script_folder = Path(__file__).parent.absolute()
        self.ui.smeeta_icon_widget.set_color([95, 255, 255])
        self.ui.text_color_widget.set_color([0, 0, 255])
        self.guirestore(QtCore.QSettings(os.path.join(self.script_folder, 'saved_settings.ini'), QtCore.QSettings.IniFormat))


        user32 = ctypes.windll.user32
        self.screen_capture:WindowCapture = WindowCapture('Warframe', ( user32.GetSystemMetrics(0) , user32.GetSystemMetrics(1) ), self.ui )
        self.overlay:Overlay = Overlay()
        self.overlay.show()
        self.window_data:WindowData = WindowData(self.ui)
        self.monitor:Monitor = Monitor(self)
        self.hotkey:Hotkey = Hotkey(self)
        self.charm_history:CharmHistory = CharmHistory(self)
        self.update_charm_history_labels()

        self.check_for_updates()

        self.ui.time_120_radio_button.toggled.connect(self.window_data.update)
        self.ui.drop_chance_booster_checkbox.stateChanged.connect(self.window_data.update)
        self.ui.drop_booster_checkbox.stateChanged.connect(self.window_data.update)
        self.ui.drop_booster_checkbox_2.stateChanged.connect(self.window_data.update)
        self.ui.bless_booster_checkbox.stateChanged.connect(self.window_data.update)
        self.ui.dark_sector_booster_spinner.valueChanged.connect(self.window_data.update)
        self.ui.ui_scale_spinner.valueChanged.connect(self.window_data.update)
        self.ui.template_match_slider.valueChanged.connect(self.window_data.update)
        self.ui.refresh_rate_slider.valueChanged.connect(self.window_data.update)
        self.ui.smeeta_icon_widget.update.connect(self.window_data.update)
        self.ui.text_color_widget.update.connect(self.window_data.update)
        self.ui.play_all_proc_sounds_checkbox.stateChanged.connect(self.window_data.update)

        self.ui.override_hotkey_button.clicked.connect(self.hotkey.hotkey_next_keypress)
        self.ui.start_button.clicked.connect(self.monitor.start_scanning)
        self.ui.stop_button.clicked.connect(self.monitor.stop_scanning)

        self.ui.ui_scale_spinner.valueChanged.connect(self.monitor.screen_scanner.update_ui_scale)
        self.ui.text_color_widget.update.connect(self.monitor.screen_scanner.update_text_color)

        self.ui.reset_text_color_default_button.clicked.connect(lambda: self.ui.text_color_widget.set_color([0, 0, 255]))
        self.ui.reset_icon_color_default_button.clicked.connect(lambda: self.ui.smeeta_icon_widget.set_color([95, 255, 255]))
        self.ui.smeeta_icon_widget.clicked.connect(lambda : self.select_color(self.ui.smeeta_icon_widget))
        self.ui.text_color_widget.clicked.connect(lambda : self.select_color(self.ui.text_color_widget))
        self.ui.overlay_text_size_slider.valueChanged.connect(lambda : self.overlay.set_text_size(13*(1+self.ui.overlay_text_size_slider.value()/100)))

        self.ui.analyze_ee_log_button.clicked.connect(self.plot_logs)
        self.ui.test_bounds_button.clicked.connect(self.display_detection_area)
        self.ui.test_icon_threshold_button.clicked.connect(self.display_icon_filter)
        self.ui.test_text_threshold_button.clicked.connect(self.display_text_filter)

        #self.ui.ui_scale_spinner.lineEdit().setReadOnly(True)

        # make a list of ui elements to disable when running
        self.ui_elements = [    self.ui.start_button,
                                self.ui.overlay_checkbox,
                                self.ui.sounds_checkbox,
                                self.ui.ui_scale_spinner,
                                self.ui.enable_affinity_scanner_checkbox,
                                self.ui.smeeta_icon_widget,
                                self.ui.text_color_widget,
                                self.ui.reset_icon_color_default_button,
                                self.ui.reset_text_color_default_button,
                                self.ui.label_5,
                                self.ui.label_6,
                                self.ui.time_120_radio_button,
                                self.ui.time_156_radio_button,
                                self.ui.test_bounds_button,
                            ]
        self.image_label_list = [self.ui.image_label_1,self.ui.image_label_2,self.ui.image_label_3,self.ui.image_label_4,self.ui.image_label_5]
        self.label_image_label_list = [self.ui.label_image_label_1,self.ui.label_image_label_2,self.ui.label_image_label_3,self.ui.label_image_label_4,self.ui.label_image_label_5]

        self.create_files()

        self.debug_image = PaintPicture(self)

        
    def load_ui(self):
        loader = QUiLoader()
        loader.registerCustomWidget(StickySpinBox)
        loader.registerCustomWidget(ColorWidget)
        path = os.fspath(Path(__file__).resolve().parent / "form.ui")
        ui_file = QFile(path)
        ui_file.open(QFile.ReadOnly)
        self.ui = loader.load(ui_file, self)
        ui_file.close()

    def create_files(self):
        if not os.path.isfile(os.path.join(self.script_folder,"user_ExportRegions.json")):
            os.rename(os.path.join(self.script_folder,'ExportRegions.json'),os.path.join(self.script_folder,'user_ExportRegions.json'))

    def closeEvent(self, arg):
        self.guisave(QtCore.QSettings(os.path.join(self.script_folder, 'saved_settings.ini'), QtCore.QSettings.IniFormat))
        self.monitor.monitor_game = False
        self.monitor.stop_scanning()
        sys.exit(0)
    
    def check_for_updates(self):
        self.latest_version_json = json.loads(requests.get(version_link).text)
        with open(os.path.join(self.script_folder,"version.json"), 'r') as f:
            version_json = json.load(f)
        if self.latest_version_json["version"] != version_json["version"]:
            self.ui.version_label.setText('''<a href='https://github.com/A-DYB/smeeta-tracker-2/releases'>Updates available!</a>''')
            self.ui.version_label.setOpenExternalLinks(True)
        else:
            self.ui.version_label.setText("Version:%s - No updates available"%version_json["version"])

    def guisave(self, settings:QtCore.QSettings):
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
                value = obj.isChecked()
                settings.setValue(name, value)
                
            if isinstance(obj, QSpinBox):
                name  = obj.objectName()
                value = obj.value()           
                settings.setValue(name, value)

            if isinstance(obj, QDoubleSpinBox):
                name  = obj.objectName()
                value = obj.value()           
                settings.setValue(name, value)

            if isinstance(obj, QSlider):
                name  = obj.objectName()
                value = obj.value()           
                settings.setValue(name, value)

            if isinstance(obj, ColorWidget):
                name  = obj.objectName()
                value = obj.color_hsv          
                settings.setValue(name, value)

    def guirestore(self, settings:QtCore.QSettings):
        self.move(settings.value('pos', QtCore.QPoint(60, 60)))
        for name, obj in inspect.getmembers(self.ui):
            if isinstance(obj, QComboBox):
                index = obj.currentIndex()
                name = obj.objectName()

                value = (settings.value(name))

                if value == "":
                    continue

                index = obj.findText(value)

                if index == -1:  # add to list if not found
                    obj.insertItems(0, [value])
                    index = obj.findText(value)
                    obj.setCurrentIndex(index)
                else:
                    obj.setCurrentIndex(index)  # preselect a combobox value by index
                    
            if isinstance(obj, QCheckBox):
                name = obj.objectName()
                value = settings.value(name)
                if value != None:
                    obj.setChecked(str2bool(value))

            if isinstance(obj, QRadioButton):
                name = obj.objectName()
                value = settings.value(name)
                if value != None:
                    obj.setChecked(str2bool(value))
            
            if isinstance(obj, QSlider):
                name = obj.objectName()
                value = settings.value(name)  
                if value != None:           
                    obj. setValue(int(value))

            if isinstance(obj, QSpinBox):
                name = obj.objectName()
                value = settings.value(name)  
                if value != None:
                    obj. setValue(int(value))

            if isinstance(obj, QDoubleSpinBox):
                name = obj.objectName()
                value = settings.value(name)  
                if value != None:
                    obj. setValue(float(value))

            if isinstance(obj, ColorWidget):
                name = obj.objectName()
                value = settings.value(name)
                if value != None:
                    obj.set_color(value)

    def select_color(self, obj:'ColorWidget'):
        if not self.screen_capture.is_window():
            return
        self.paint = PaintPicture(self)
        self.paint.show_image(self.screen_capture.get_screenshot(), obj)
    
    def on_data_ready(self):
        self.overlay.scan_label_group.reset_text()
        self.overlay.log_label_group.reset_text()

        for i in range(len(self.image_label_list)):
            if i < len(self.window_data.scan_display_data.image_list):
                self.image_label_list[i].setPixmap(self.window_data.scan_display_data.image_list[i])
                self.label_image_label_list[i].setText(self.window_data.scan_display_data.text_list[i])
            else:
                self.image_label_list[i].clear()
                self.label_image_label_list[i].setText("")

        next_affinity_expiry_unix = None
        next_affinity_expiry_unix = self.monitor.screen_scanner.get_next_expiry_unix()
        if next_affinity_expiry_unix:
            next_affinity_expiry_s = next_affinity_expiry_unix - time.time()
            active_procs = len(self.monitor.screen_scanner.affinity_proc_list)
            remaining_chances = int((round(next_affinity_expiry_s,1)+6)/27)

            self.overlay.scan_label_group.add_text(f'Active: {active_procs}')
            self.overlay.scan_label_group.add_text(f'Next Expiry: {int(next_affinity_expiry_s)}', color="rgb(148, 255, 119)", bold=True)
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
            ref_timestamp = max(self.monitor.log_parser.mission_start_timestamp_unix_s+1, self.monitor.screen_scanner.proc_validator.last_proc_start_unix_s, self.hotkey.smeeta_rotation_override_unix)
            self.overlay.scan_label_group.add_text(f'Charm Rotation: {(27.4-(time.time() - (ref_timestamp))%27.4):.1f}s')

        # update ui labels
        self.ui.drone_spawns_label.setText(str(self.monitor.log_parser.drone_spawns))
        self.ui.total_spawns_label.setText(str(self.monitor.log_parser.enemy_spawns))
        if not self.monitor.log_parser.in_mission:
            self.ui.mission_time_label.setText(str(datetime.timedelta(seconds=int(self.monitor.log_parser.mission_duration_s))))
        else:
            self.ui.mission_time_label.setText(str(datetime.timedelta(seconds=int(time.time() - self.monitor.log_parser.mission_start_timestamp_unix_s))))

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
            self.overlay.log_label_group.add_text(f'Arb: {mission_info_str} Exp: {(3600-(time.time()%3600))/60:.0f} mins')

        if self.ui.display_tmatch_checkbox.isChecked():
            self.overlay.scan_label_group.add_text(self.monitor.screen_scanner.template_match_status_text)

        self.update_charm_history_labels()

    def update_charm_history_labels(self):
        if self.charm_history.update:
            self.charm_history.update = False
            self.ui.affinity_count_label.setText(f'{self.charm_history.total_affinity_procs} ({self.charm_history.affinity_chance*100:.2f}%)')
            self.ui.critical_count_label.setText(f'{self.charm_history.total_critical_chance_procs} ({self.charm_history.critical_chance*100:.2f}%)')
            self.ui.energy_count_label.setText(f'{self.charm_history.total_energy_refund_procs} ({self.charm_history.energy_chance*100:.2f}%)')
            self.ui.total_mission_time_label.setText(f'{self.charm_history.total_mission_time/60:.0f} mins')
    
    def display_detection_area(self):
        if not self.monitor.screen_scanner.screen_capture.is_window():
            self.display_error(f'Cannot find a window named "Warframe".')
            return
        img = self.monitor.screen_scanner.screen_capture.get_screenshot()
        self.paint = PaintPicture(self)
        self.paint.show_image(img, None)

    def display_icon_filter(self):
        if not self.monitor.screen_scanner.screen_capture.is_window():
            self.display_error(f'Cannot find a window named "Warframe".')
            return
        img = self.monitor.screen_scanner.screen_capture.get_screenshot()
        filtered_scan = scanner.hsv_filter(img, self.ui.smeeta_icon_widget.color_hsv, h_sens=4, s_sens=60, v_scale=0.6)
        self.paint = PaintPicture(self)
        self.paint.show_image(filtered_scan, None)

    def display_text_filter(self):
        if not self.monitor.screen_scanner.screen_capture.is_window():
            self.display_error(f'Cannot find a window named "Warframe".')
            return
        img = self.monitor.screen_scanner.screen_capture.get_screenshot()
        filtered_scan = self.monitor.screen_scanner.text_hsv_filter(img, self.ui.text_color_widget.color_hsv)
        self.paint = PaintPicture(self)
        self.paint.show_image(filtered_scan, None)

    def plot_logs(self):
        self.paint = PaintPicture(self)
        self.paint.plot_logs(self)

    def show_video(self, image):
        if image is None:
            return
        if self.debug_image is None:
            self.debug_image = PaintPicture(self)
        self.debug_image.show_image_sequence(image)

    def display_error(self, error_message):
        self.dlg = QMessageBox(self)
        self.dlg.setWindowTitle("Error")
        self.dlg.setText(error_message)
        self.dlg.setIcon(QMessageBox.Warning)
        self.dlg.exec()

class WindowData():
    def __init__(self, ui) -> None:
        self.ui = ui
        self.scan_display_data = self.ScanDisplayData()
        self.update()

    def update(self):
        self.affinity_proc_duration = 120 if self.ui.time_120_radio_button.isChecked() else 156
        self.duration_multiplier = 1 if self.ui.time_120_radio_button.isChecked() else 1.3
        self.drop_chance_booster = 2 if self.ui.drop_chance_booster_checkbox.isChecked() else 1
        self.drop_booster = 2 if self.ui.drop_booster_checkbox.isChecked() else 1
        self.drop_booster2 = 2 if self.ui.drop_booster_checkbox_2.isChecked() else 1
        self.bless_booster = 1.25 if self.ui.bless_booster_checkbox.isChecked() else 1
        self.dark_sector_booster = self.ui.dark_sector_booster_spinner.value()
        self.ui_scale = self.ui.ui_scale_spinner.value()/100
        self.template_match_threshold = self.ui.template_match_slider.value()/100
        self.scanner_refresh_rate_s = 0.5+1*self.ui.refresh_rate_slider.value()/100
        self.play_all_proc_sounds = self.ui.play_all_proc_sounds_checkbox.isChecked()

        self.ui.template_match_label.setText(f'{self.ui.template_match_slider.value():.0f}%')
        self.ui.refresh_rate_label.setText(f'{self.scanner_refresh_rate_s:.1f}s')

    class ScanDisplayData():
        def __init__(self) -> None:
            self.image_list = []
            self.text_list = []
            self.image_index = 0
            self.max_images = 5

        def add_element(self, image, image_text):
            if len(self.image_list) < self.max_images:
                self.image_list.append(image)
                self.text_list.append(image_text)
                self.image_index += 1
                if self.image_index > self.max_images-1:
                    self.image_index=0
            else:
                self.image_list[self.image_index] = image
                self.text_list[self.image_index] = image_text
                self.image_index += 1
                if self.image_index > self.max_images-1:
                    self.image_index=0

class CharmHistory():
    def __init__(self, main_window:MainWindow) -> None:
        self.main_window = main_window
        self.total_affinity_procs = 0
        self.total_critical_chance_procs = 0
        self.total_energy_refund_procs = 0

        self.total_chances = 1

        self.affinity_chance = 0
        self.critical_chance = 0
        self.energy_chance = 0
        
        self.total_mission_time = 0

        self.ref_mission_start_unix_s = 0 #
        self.ref_mission_time_s = 0 #
        self.update = True

        self.load_history()

    def load_history(self):
        charm_history_file = os.path.join(self.main_window.script_folder, "charm_history.csv")
        if not os.path.isfile(charm_history_file):
            return
        df = pd.read_csv(charm_history_file)
        df['start_time_s'] = df['proc_start_timestamp_unix_s'] - df['scan_start_timestamp_unix_s']
        for start_time in df.scan_start_timestamp_unix_s.unique():
            df_f = df[(df.scan_start_timestamp_unix_s == start_time)]
            mission_duration_s = df_f['start_time_s'].max()
            self.total_mission_time += mission_duration_s

        self.ref_mission_start_unix_s = df.scan_start_timestamp_unix_s.max()
        df_f = df[(df.scan_start_timestamp_unix_s == self.ref_mission_start_unix_s)]
        self.ref_mission_time_s = df_f.proc_start_timestamp_unix_s.max()

        value_counts = df['name'].value_counts()
        if 'Affinity' in value_counts:
            self.total_affinity_procs = value_counts['Affinity']
        if 'Critical Chance' in value_counts:
            self.total_critical_chance_procs = value_counts['Critical Chance']
        if 'Energy Refund' in value_counts:
            self.total_energy_refund_procs = value_counts['Energy Refund']

        self.total_chances = max(1, self.total_mission_time // 27)

        self.affinity_chance = self.total_affinity_procs/self.total_chances
        self.critical_chance = self.total_critical_chance_procs/self.total_chances
        self.energy_chance = self.total_energy_refund_procs/self.total_chances
        self.update = True

    def new_proc(self, proc:ProcValidator.Proc):
        if proc.name == "Affinity":
            self.total_affinity_procs += 1
        elif proc.name == "Critical Chance":
            self.total_critical_chance_procs += 1
        elif proc.name == "Energy Refund":
            self.total_energy_refund_procs += 1

        if proc.scan_start_timestamp_unix_s != self.ref_mission_start_unix_s:
            self.ref_mission_start_unix_s = proc.scan_start_timestamp_unix_s
            ref_mission_time_s = (proc.start_timestamp_unix_s - proc.scan_start_timestamp_unix_s)
            self.total_mission_time += ref_mission_time_s
            self.ref_mission_time_s = ref_mission_time_s
        else:
            new_ref_mission_time_s = (proc.start_timestamp_unix_s - proc.scan_start_timestamp_unix_s)
            self.total_mission_time -= self.ref_mission_time_s
            self.total_mission_time += new_ref_mission_time_s

        self.total_chances = max(1, self.total_mission_time // 27)

        self.affinity_chance = self.total_affinity_procs/self.total_chances
        self.critical_chance = self.total_critical_chance_procs/self.total_chances
        self.energy_chance = self.total_energy_refund_procs/self.total_chances
        self.update = True

class Hotkey():
    def __init__(self, main_window) -> None:
        self.main_window = main_window
        self.ui = main_window.ui
        self.save_next_hotkey = False
        self.smeeta_rotation_override_unix = 0

        self.hotkey = keyboard.KeyCode.from_char('k')
        hotkey_file = os.path.join(self.main_window.script_folder, 'hotkey.pkl')
        if os.path.isfile(hotkey_file):
            with open(os.path.join(self.main_window.script_folder, 'hotkey.pkl'), 'rb') as f:
                self.hotkey = pickle.load(f)
        self.ui.hotkey_label.setText(self.hotkey.char)
        listener = keyboard.Listener(on_press=self.on_press)
        listener.start()

    def on_press(self, key):
        try:
            if self.save_next_hotkey:
                self.save_hotkey(key)
                self.save_next_hotkey = False
                return
            if key.char == self.hotkey.char:
                print(f'Key {key.char} was pressed, resetting charm rotation')
                self.smeeta_rotation_override_unix = time.time()
                self.main_window.on_data_ready()
        except AttributeError:
            pass

    def save_hotkey(self, key):
        with open(os.path.join(self.main_window.script_folder, 'hotkey.pkl'), 'wb') as outfile:
            pickle.dump(key, outfile)
        self.hotkey = key
        self.ui.hotkey_label.setText(self.hotkey.char)
    
    def hotkey_next_keypress(self):
        self.ui.hotkey_label.setText("Press Key")

        self.ui.override_hotkey_button.setStyleSheet("")

        self.save_next_hotkey = True

class Overlay(QtWidgets.QDialog):
    def __init__(self):
        super(Overlay, self).__init__()
        self.setWindowIcon(QtGui.QIcon(':/icons/charm.png'))

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
            for i in range(6):
                self.label_list.append( QtWidgets.QLabel(''))
                self.label_list[i].setFont(QtGui.QFont('Arial', 13))
                self.label_list[i].setStyleSheet("color : white")
                layout.addWidget(self.label_list[i])

        def add_text(self, text, color="white", bold=False):
            if GetWindowText(GetForegroundWindow()) == 'Warframe':
                if self.occupied_labels >= len(self.label_list):
                    self.reset_text()
                self.label_list[self.occupied_labels].setText(text)

                fontweight = "bold" if bold else "normal"
                stylesheet = f'color : {color}; font-weight: {fontweight}'
                self.label_list[self.occupied_labels].setStyleSheet(stylesheet)

                self.occupied_labels += 1

        def reset_text(self):
            for lbl in self.label_list:
                lbl.setText('')
            self.count +=1
            if self.count >5:
                self.count=0
            self.occupied_labels = 0

        def set_text_size(self, size):
            for lbl in self.label_list:
                lbl.setFont(QtGui.QFont('Arial', int(size)))

class PaintPicture(QtWidgets.QDialog):
    def __init__(self, parent=None):
        super(PaintPicture, self).__init__()

        layout = QtWidgets.QVBoxLayout(self)
        self.setWindowFlags(QtCore.Qt.Dialog | QtCore.Qt.MSWindowsFixedSizeDialogHint | QtCore.Qt.WindowStaysOnTopHint)
        self.setLayout(layout)
        self.color_widget = None
        self.image_sequence_label = None

    def show_image(self, cv_img, color_widget:'ColorWidget'):
        self.image = cv_img
        if len(cv_img.shape) == 3:
            self.image_hsv = cv2.cvtColor(cv_img, cv2.COLOR_BGR2HSV)
        self.color_widget = color_widget

        x, y = cv_img.shape[1], cv_img.shape[0]

        self.imageLabel = QtWidgets.QLabel()
        self.imageLabel.setGeometry(0, 0, x, y)

        pixmap = convert_cv_qt(cv_img)
        self.imageLabel.setPixmap(pixmap)

        layout = self.layout()
        layout.addWidget(self.imageLabel)
        self.show()
        self.imageLabel.move(0,0)

    def show_image_sequence(self, image):
        self.image = image
        if len(image.shape) == 3:
            self.image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        x, y = image.shape[1], image.shape[0]

        add_label = False
        if self.image_sequence_label is None:
            add_label=True
            self.image_sequence_label = QtWidgets.QLabel()
            self.image_sequence_label.setGeometry(0, 0, x, y)

        pixmap = convert_cv_qt(image)
        self.image_sequence_label.setPixmap(pixmap)

        if add_label:
            layout = self.layout()
            layout.addWidget(self.image_sequence_label)
            self.image_sequence_label.move(0,0)
        self.show()

    def remove_all_widgets(self):
        layout = self.layout()
        for i in reversed(range(layout.count())): 
            layout.itemAt(i).widget().setParent(None)

    def mousePressEvent(self, event: QMouseEvent) -> None:
        y,x = int(event.position().y()), int(event.position().x())

        if self.color_widget is not None and y < self.image_hsv.shape[0] and x < self.image_hsv.shape[1]:
            self.color_widget.set_color(self.image_hsv[y,x])
            self.color_widget = None
            self.accept()
        return super().mousePressEvent(event)
    
    def plot_logs(self, main_window:MainWindow):
        self.sc = MplCanvas(self, width=7, height=7, dpi=100)

        df = main_window.monitor.log_parser.parse_arbitration_logs()
        if df is None:
            main_window.display_error(f'No mission data to analyze.')
            print(f'No mission data')
            return

        sns.lineplot(data=df, x='mission_time_minutes', y='drones_per_hour', ax=self.sc.axs[3], errorbar=None)
        sns.lineplot(data=df, x='mission_time_minutes', y='drones_per_enemy', ax=self.sc.axs[1], errorbar=None)
        sns.lineplot(data=df, x='mission_time_minutes', y='enemy_count', ax=self.sc.axs[0], errorbar=None)
        sns.lineplot(data=df, x='mission_time_minutes', y='drone_count', ax=self.sc.axs[2], errorbar=None)

        self.sc.axs[0].set_title('Enemy spawns')
        self.sc.axs[2].set_title('Drone spawns')
        self.sc.axs[1].set_title('Drones per enemy')
        self.sc.axs[3].set_title('Drones per hour')

        df.sort_values(by='mission_time_minutes', inplace=True)
        vitus_chance = main_window.monitor.log_parser.get_vitus_essence_chance()
        df['boost'] = np.array([vitus_chance]*len(df.index))
        df['drone_count_diff'] = np.diff(df.drone_count.to_numpy(), prepend=0)

        charm_history_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'charm_history.csv')
        if os.path.isfile(charm_history_file):
            df_s = pd.read_csv(charm_history_file)
            if not df_s.empty:
                df_s_filt = df_s[(df_s.name == "Affinity")]
                smeeta_proc_timestamps_unix_s = df_s_filt['proc_start_timestamp_unix_s'].to_numpy()
                affinity_duration = df_s_filt['proc_duration_s'].iloc[0]

                mission_start_time_unix_s = df.timestamp_unix_s.min()
                mission_end_time_unix_s = df.timestamp_unix_s.max()
                mission_smeeta_proc_timestamps_unix_s = smeeta_proc_timestamps_unix_s[np.where((smeeta_proc_timestamps_unix_s>mission_start_time_unix_s) & (smeeta_proc_timestamps_unix_s < mission_end_time_unix_s))]
                for smeeta_proc_timestamp in mission_smeeta_proc_timestamps_unix_s:
                    self.sc.axs[2].axvspan((smeeta_proc_timestamp - mission_start_time_unix_s)/60, (smeeta_proc_timestamp - mission_start_time_unix_s + affinity_duration)/60, alpha=0.5, color='green')
                    df.loc[ (df.timestamp_unix_s > smeeta_proc_timestamp) & (df.timestamp_unix_s < smeeta_proc_timestamp + affinity_duration), 'boost'] *= 2
                self.sc.axs[2].legend(['Smeeta proc'])
                leg = self.sc.axs[2].get_legend()
                leg.legend_handles[0].set_color('green')   

        total_vitus_essence = np.sum( np.multiply( df.boost.to_numpy(), df.drone_count_diff.to_numpy() ) )

        # get last mission
        mission_info_str = main_window.monitor.log_parser.get_node_info_string(main_window.monitor.log_parser.recently_played_mission)

        self.sc.fig.suptitle(f'{mission_info_str}\nDrones: {df.drone_count.max()}\nAvg VE Drops: {total_vitus_essence:.0f}; Avg Boost: {total_vitus_essence/(max(1,df.drone_count.max())*0.06):.2f}')
        self.sc.fig.tight_layout()
        
        layout = self.layout()
        layout.addWidget(self.sc)
        self.show()

class MplCanvas(FigureCanvasQTAgg):
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        self.fig.add_subplot(221)
        self.fig.add_subplot(222)
        self.fig.add_subplot(223)
        self.fig.add_subplot(224)
        self.axs = self.fig.axes
        super(MplCanvas, self).__init__(self.fig)

class StickySpinBox(QSpinBox):
    def __init__(self, parent=None, *args):
        super().__init__(parent, *args)
        self.setSingleStep(50)
        self.valueChanged.connect(self.roundValue)

    def roundValue(self):
        value = self.value()
        rounded_value = int(100*round(value/100 * 2) / 2)
        if rounded_value != value:
            self.setValue(rounded_value)

class ColorWidget(QFrame):
    clicked = Signal()
    update = Signal()
    def __init__(self, parent=None, *args):
        super().__init__(parent, *args)
        self.color_hsv = [0,0,0]
    
    def mousePressEvent(self, event: QMouseEvent) -> None:
        self.clicked.emit()
        return super().mousePressEvent(event)
    
    def set_color(self, value:list):
        if len(value) != 3:
            print(f'Invalid argument to ColorWidget set_color')
            return
        if type(value) == np.ndarray:
            value = list(value)
        
        for i, elem in enumerate(value):
            if not is_float(elem):
                print(f'Failed to cast {elem} to int')
                break
            value[i] = int(value[i])
        else:
            stylesheet_str = f'ColorWidget#{self.objectName()} {{ border: 1 solid hsv(0,0,0); background-color: hsv({int(value[0])*2:d},{int(value[1]):d},{int(value[2]):d}); }}\
                                ColorWidget#{self.objectName()}:hover {{background-color: hsv({int(value[0])*2:d},{int(value[1]):d},{int(value[2]*0.7):d});}}'
            self.setStyleSheet(stylesheet_str)
            self.color_hsv = value
            self.update.emit()

def convert_cv_qt(cv_img):
    """Convert from an opencv image to QPixmap"""
    rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
    h, w, ch = rgb_image.shape
    bytes_per_line = ch * w
    convert_to_Qt_format = QtGui.QImage(rgb_image.data, w, h, bytes_per_line, QtGui.QImage.Format_RGB888)
    #p = convert_to_Qt_format.scaled(w, h, QtCore.Qt.KeepAspectRatio)
    return QtGui.QPixmap.fromImage(convert_to_Qt_format)

def is_float(element: any) -> bool:
    if element is None: 
        return False
    try:
        float(element)
        return True
    except ValueError:
        return False

def str2bool(v):
  return v.lower() in ("yes", "true", "t", "1")

if __name__ == "__main__":
    QtCore.QCoreApplication.setAttribute(QtCore.Qt.AA_ShareOpenGLContexts)
    app = QApplication([])
    app.setStyle('Fusion')
    widget = MainWindow()
    widget.show()

    sys.exit(app.exec())



