from EeLogParser import EeLogParser
from scanner import ScreenScanner

import threading
import time
import os
from playsound import playsound
import win32gui
import PySide6.QtCore as QtCore

class Monitor():
    def __init__(self, parent) -> None:
        self.parent_window=parent
        self.clear_terminal = lambda: os.system('cls')

        self.monitor_game = False
        self.log_parser = EeLogParser(self.parent_window)
        self.screen_scanner = ScreenScanner(self.parent_window)
        self.thread_list=[]
        self.qthread_list = []

    def start_scanning(self):
        if win32gui.FindWindow(None, 'Warframe')is None:
            print('Cannot find Warframe window')
            return 
        
        self.log_parser.reset()
        self.screen_scanner = ScreenScanner(self.parent_window)
        self.monitor_game = True
        self.parent_window.disable_setting_buttons()

        if self.parent_window.ui.enable_affinity_scanner_checkbox.isChecked():
            sst = self.ScreenScannerThread(self.parent_window, self.screen_scanner, self.monitor_game)
            sst.update_gui.connect(self.parent_window.on_data_ready)
            sst.start()
            self.qthread_list.append(sst)
            self.screen_scanner.sound_queue.append('starting_scan.mp3')

            uot = self.UpdateOverlayThread(self.parent_window, self.screen_scanner, self.log_parser, self.monitor_game)
            uot.update_gui.connect(self.parent_window.on_data_ready)
            uot.start()
            self.qthread_list.append(uot)

        if self.parent_window.ui.sounds_checkbox.isChecked():
            x3 = threading.Thread(target=self.sound_thread, daemon=True )
            x3.start()
            self.thread_list.append(x3)
            

    def stop_scanning(self):
        for th in self.qthread_list:
            th.monitor_game=False
        self.monitor_game = False
        thread_monitor = threading.Thread(target=self.monitor_threads, daemon=True )
        thread_monitor.start()
        thread_monitor.join()

    def monitor_threads(self):
        while len(self.thread_list) > 0 and len(self.qthread_list)>0:
            time.sleep(1)
        self.parent_window.overlay.scan_label_group.reset_text()
        self.parent_window.overlay.log_label_group.reset_text()
        self.parent_window.enable_setting_buttons()

    def sound_thread(self):
        while self.monitor_game:
            while( len(self.screen_scanner.sound_queue) > 0 and self.monitor_game):
                sound_file = os.path.join(self.parent_window.script_folder,'Sounds', self.screen_scanner.sound_queue[0] )
                if os.path.isfile(sound_file):
                    playsound(sound_file, block=True)
                else:
                    print("Sound file does not exist: ", sound_file)
                if len(self.screen_scanner.sound_queue) > 0:
                    self.screen_scanner.sound_queue.popleft()
            time.sleep(3)
        self.thread_list.pop()
    
    class ScreenScannerThread(QtCore.QThread):
        update_gui = QtCore.Signal(object)
        def __init__(self, parent, screen_scanner, monitor_game) -> None:
            QtCore.QThread.__init__(self, parent)
            self.screen_scanner = screen_scanner
            self.monitor_game = monitor_game

        def run(self):
            while(self.monitor_game):
                self.screen_scanner.scan_match_template()
                #self.update_gui.emit(0)
                time.sleep(self.screen_scanner.refresh_rate_s)          

    class UpdateOverlayThread(QtCore.QThread):
        update_gui = QtCore.Signal(object)
        def __init__(self, parent, screen_scanner, log_parser, monitor_game) -> None:
            QtCore.QThread.__init__(self, parent)
            self.screen_scanner = screen_scanner
            self.monitor_game = monitor_game
            self.log_parser = log_parser
            self.ee_log_size = 0

        def run(self):
            while(self.monitor_game):
                # scan logs
                ee_log_size = os.stat(self.log_parser.ee_log_path).st_size
                if ee_log_size > self.ee_log_size:
                    try:
                        self.log_parser.parse_file()
                        self.ee_log_size = ee_log_size
                    except Exception as e:
                        print(f'Failed to read ee log: {e}')

                next_affinity_expiry_unix = None
                next_affinity_expiry_unix = self.screen_scanner.get_next_expiry_unix()
                self.update_gui.emit(0)
                if next_affinity_expiry_unix:
                    time.sleep(min(1, (next_affinity_expiry_unix - time.time())%1))
                else:
                    ref_timestamp = max(self.log_parser.mission_start_timestamp_unix_s+1, self.screen_scanner.proc_validator.last_proc_reference_timestamp_unix_s)
                    charm_rotation = (27.4-(time.time() - (ref_timestamp))%27.4)
                    time.sleep(charm_rotation%1)
            print('monitor_game was set false! exiting overlay update thread')