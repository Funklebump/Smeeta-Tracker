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
        self.scan_start_timestamp_unix_s = 0

    def start_scanning(self):
        if not self.screen_scanner.screen_capture.is_window():
            self.parent_window.display_error(f'Cannot find a window named "Warframe".')
            return 
        
        self.scan_start_timestamp_unix_s = time.time()
        
        self.log_parser.reset()
        self.screen_scanner = ScreenScanner(self.parent_window)
        self.monitor_game = True
        for elem in self.parent_window.ui_elements:
            elem.setEnabled(False)

        if self.parent_window.ui.enable_affinity_scanner_checkbox.isChecked():
            sst = self.ScreenScannerThread(self.parent_window, self.screen_scanner, self.monitor_game)
            sst.update_gui.connect(self.parent_window.on_data_ready)
            sst.update_image.connect(lambda: self.parent_window.show_video(sst.screen_scanner.debug_image))
            sst.exit_.connect(self.stop_scanning)
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

        for elem in self.parent_window.ui_elements:
            elem.setEnabled(True)

    def monitor_threads(self):
        while len(self.thread_list) > 0 and len(self.qthread_list)>0:
            time.sleep(1)
        self.parent_window.overlay.scan_label_group.reset_text()
        self.parent_window.overlay.log_label_group.reset_text()

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
            time.sleep(1)
        self.thread_list.pop()
    
    class ScreenScannerThread(QtCore.QThread):
        update_image = QtCore.Signal()
        update_gui = QtCore.Signal(object)
        exit_ = QtCore.Signal()
        def __init__(self, parent, screen_scanner, monitor_game) -> None:
            QtCore.QThread.__init__(self, parent)
            self.screen_scanner = screen_scanner
            self.monitor_game = monitor_game
            self.window = parent

        def run(self):
            while(self.monitor_game):
                if self.screen_scanner.exit_:
                    self.exit_.emit()
                    break
                self.screen_scanner.find_charm_proc()
                self.screen_scanner.proc_validator.remove_expired_procs()
                #self.update_image.emit()
                time.sleep(self.window.window_data.scanner_refresh_rate_s)          

    class UpdateOverlayThread(QtCore.QThread):
        update_gui = QtCore.Signal(object)
        def __init__(self, parent, screen_scanner:ScreenScanner, log_parser:EeLogParser, monitor_game:bool) -> None:
            QtCore.QThread.__init__(self, parent)
            self.parent = parent
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
                        self.log_parser.parse_latest_logs()
                        self.ee_log_size = ee_log_size
                    except Exception as e:
                        print(f'Failed to read ee log: {e}')

                next_affinity_expiry_unix = None
                next_affinity_expiry_unix = self.screen_scanner.get_next_expiry_unix()
                self.update_gui.emit(0)
                if next_affinity_expiry_unix:
                    time.sleep(min(1, (next_affinity_expiry_unix - time.time())%1))
                else:
                    ref_timestamp = max(self.log_parser.mission_start_timestamp_unix_s+1, self.screen_scanner.proc_validator.last_proc_start_unix_s, self.parent.hotkey.smeeta_rotation_override_unix)
                    charm_rotation = (27.4-(time.time() - (ref_timestamp))%27.4)
                    time.sleep(charm_rotation%1)
            print('monitor_game was set false! exiting overlay update thread')