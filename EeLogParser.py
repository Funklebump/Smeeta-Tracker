# This Python file uses the following encoding: utf-8

import time
from os import SEEK_END
import io
from lz.reversal import reverse
import logging
import os
from win32com.shell import shell, shellcon
import calendar
import datetime
import matplotlib.pyplot as plt
import numpy as np
import csv
import re
import json

class EeLogParser:
    def __init__(self, max_proc_time, ui):
        self.ui = ui
        self.ee_log_path = os.path.join(shell.SHGetFolderPath(0, shellcon.CSIDL_LOCAL_APPDATA, None, 0), 'Warframe\ee.log')
        self.global_time = 0
        self.sync_time()
        self.mission_start_time = 0
        self.mission_end_time = 0
        self.scan_log_time=0
        self.previous_log_time=0
        self.latest_log_time=0
        self.initial_scan=True
        self.in_mission=False
        self.mission_time=0
        self.max_proc_time=max_proc_time
        self.first_scan=True
        self.current_arbitration=""

        self.mission_state_found=False
        self.drone_spawns = 0
        self.total_spawns = 0

        self.status_text=''
        self.last_spawn_time=0
        self.time_regex = re.compile('\d*[.]\d{3}')

    def sync_time(self):
        with open(self.ee_log_path) as log_file:
            for line in log_file:
                #find time alignment
                if "Sys [Diag]: Current time:" in line:
                    res = line.split("[")[2]
                    res = res.split()
                    s = res[3]+"-"+res[2]+"-"+res[5].split("]")[0]+":"+res[4]
                    timestruct = time.strptime(s, "%d-%b-%Y:%H:%M:%S")
                    self.global_time = calendar.timegm(timestruct) - float(line.split(" ")[0])
                    break

    def parse_file(self):
        i=-1
        drone_list=[]
        event_list=[]
        found_mission_end=False
        with open(self.ee_log_path) as log_file:
            for line in reverse(log_file, batch_size=io.DEFAULT_BUFFER_SIZE):
                if i ==-1:
                    # skip the first line since it can be in the process of being written to
                    i+=1
                    continue
                time_stamp_str = line.split(" ")[0]
                reg_res = self.time_regex.match(time_stamp_str)
                if reg_res is None:
                    logging.info("Could not find timestamp on current line, continuing")
                    continue
                if isfloat(time_stamp_str):
                    if i == 0:
                        self.latest_log_time = float(time_stamp_str)
                    self.scan_log_time = float(time_stamp_str)
                else:
                    continue
                if self.scan_log_time <= self.previous_log_time:
                    #We are up to date on latest logs
                    break
                #mission start state
                if "GameRulesImpl::StartRound()" in line or 'Game [Info]: OnStateStarted, mission type' in line:
                    if not found_mission_end:
                        if not self.first_scan:
                            self.drone_spawns=0
                            self.total_spawns=0
                            self.last_spawn_time=0
                        self.in_mission=True
                    dt=datetime.datetime.fromtimestamp(self.scan_log_time+self.global_time).strftime('%Y-%m-%d %H:%M:%S')
                    #logging.info('Mission started at time: %s'%(dt))
                    event_list.append('Mission started at time: %s'%(dt))
                    self.mission_start_time = self.global_time + self.scan_log_time
                    break
                elif 'Game [Info]: CommitInventoryChangesToDB' in line:
                    found_mission_end=True
                    #logging.info('Not in mission')
                    event_list.append('Not in mission')
                    self.mission_end_time = self.global_time + self.scan_log_time
                    self.in_mission=False
                    with open("solNodes.json") as f:
                        map_info = json.load(f)
                    if self.drone_count > map_info[self.current_misison]["personal_best"]:
                        map_info[self.current_misison]["personal_best_dph"] = self.drone_count
                        with open('solNodes.json', 'w') as outfile:
                            json.dump(map_info, outfile)

                #find drone spawns
                if "OnAgentCreated /Npc/CorpusEliteShieldDroneAgent" in line:
                    #drone_list.append(self.global_time+self.scan_log_time)
                    date_string = datetime.datetime.fromtimestamp(int(self.global_time+self.scan_log_time)).strftime('%Y-%m-%d %H:%M:%S')
                    event_list.append("Arbitration drone spawned at time: %s"%(date_string))
                    self.drone_spawns+=1
                    self.last_spawn_time=int(self.global_time+self.scan_log_time)
                if "OnAgentCreated" in line and "/Npc/AutoTurretAgentShipRemaster" not in line:
                    self.total_spawns+=1
                if 'Script [Info]: Arbitration.lua: Destroying CorpusEliteShieldDroneAvatar' in line:
                    self.drone_spawns-=1
                    date_string = datetime.datetime.fromtimestamp(int(self.global_time+self.scan_log_time)).strftime('%Y-%m-%d %H:%M:%S')
                    event_list.append("Arbitration drone despawned: %s"%(date_string))
                if not self.first_scan and "Script [Info]: Background.lua: EliteAlertMission at " in line:
                    with open("solNodes.json") as f:
                        map_info = json.load(f)
                    if "SolNode" in line:
                        node = (re.search(r'SolNode[\d]+', line)).group(0)
                    else:
                        node = (re.search(r'ClanNode[\d]+', line)).group(0)
                    self.current_arbitration = "%s %s %s (PB: %d drones/hr)"%(map_info[node]['value'],map_info[node]['enemy'],map_info[node]['type'],map_info[node]['personal_best_dph'])
                if not self.first_scan and "Script [Info]: ThemedSquadOverlay.lua: Host loading {\"name\":" in line:
                    if "SolNode" in line:
                        self.current_misison = (re.search(r'SolNode[\d]+', line)).group(0)
                    else:
                        self.current_misison = (re.search(r'ClanNode[\d]+', line)).group(0)
                i+=1
        event_list.reverse()
        for elem in event_list:
            logging.info(elem)
        self.previous_log_time = self.latest_log_time

        if self.in_mission:
            self.mission_time = self.latest_log_time-(self.mission_start_time-self.global_time)
        else:
            self.mission_time = self.mission_end_time-self.mission_start_time
        if self.first_scan:
            self.search_arbitration()
        self.first_scan=False
        return self.drone_spawns

    def search_arbitration(self):
        with open(self.ee_log_path) as log_file:
            for line in reverse(log_file, batch_size=io.DEFAULT_BUFFER_SIZE):
                if "Script [Info]: Background.lua: EliteAlertMission at " in line:
                    with open("solNodes.json") as f:
                        map_info = json.load(f)
                    if "SolNode" in line:
                        node = (re.search(r'SolNode[\d]+', line)).group(0)
                    else:
                        node = (re.search(r'ClanNode[\d]+', line)).group(0)
                    self.current_arbitration = "%s %s %s"%(map_info[node]['value'],map_info[node]['enemy'],map_info[node]['type'])
                    break
        # get most recent mission
        with open(self.ee_log_path) as log_file:
            for line in reverse(log_file, batch_size=io.DEFAULT_BUFFER_SIZE):
                if not self.first_scan and "Script [Info]: ThemedSquadOverlay.lua: Host loading {\"name\":" in line:
                    if "SolNode" in line:
                        self.current_misison = (re.search(r'SolNode[\d]+', line)).group(0)
                    else:
                        self.current_misison = (re.search(r'ClanNode[\d]+', line)).group(0)

    def plot_logs(self):
        fig, axs = plt.subplots(2,2)
        #fig.suptitle('Vertically stacked subplots')

        axs[0][0].set_ylabel('Count')
        axs[0][0].set_title('Enemy spawns')
        axs[1][0].set_ylabel('Count')
        axs[1][0].set_title('Drone spawns')

        axs[0][1].set_title('Normalized spawns')
        axs[1][1].set_title('Drone Rate (drones per enemy)')
        axs[1][1].set_ylabel('Percent')

        enemy_spawn_times=[]
        enemy_spawn_count=[]
        drone_spawn_times=[]
        drone_spawn_count=[]
        start_time=0
        en_count=0
        drone_count=0
        with open(self.ee_log_path) as log_file:
            for line in reverse(log_file, batch_size=io.DEFAULT_BUFFER_SIZE):
                if isfloat(line.split(" ")[0]):
                    t_val = float(line.split(" ")[0])
                if "GameRulesImpl::StartRound()" in line or 'Game [Info]: OnStateStarted, mission type' in line:
                    start_time=t_val
                    print(start_time)
                    break
                elif 'Game [Info]: CommitInventoryChangesToDB' in line:
                    mission_end_time = self.global_time + t_val
                if "OnAgentCreated /Npc/CorpusEliteShieldDroneAgent" in line:
                    drone_spawn_times.append(t_val)
                    drone_count+=1
                if "OnAgentCreated" in line and "/Npc/AutoTurretAgentShipRemaster" not in line:
                    enemy_spawn_times.append(t_val)
                    en_count+=1
                if 'Script [Info]: Arbitration.lua: Destroying CorpusEliteShieldDroneAvatar' in line:
                    drone_count-=1
                    if len(drone_spawn_times)>0 and len(enemy_spawn_count)>0:
                        drone_spawn_times.pop(-1)
                        enemy_spawn_count.pop(-1)
        drone_spawn_count = np.array(range(1,len(drone_spawn_times)+1))
        enemy_spawn_count = np.array(range(1,len(enemy_spawn_times)+1))
        enemy_spawn_times.reverse()
        drone_spawn_times.reverse()

        enemy_spawn_times = np.array([f-start_time for f in enemy_spawn_times])
        drone_spawn_times = np.array([f-start_time for f in drone_spawn_times])

        if len(enemy_spawn_times)>0 and len(drone_spawn_times)>0:
            if len(drone_spawn_times)>0:
                drone_rate=[]
                j=0
                for i, elem in enumerate(enemy_spawn_count):
                    if enemy_spawn_times[i] > drone_spawn_times[j]:
                        j+=1
                    if j>=len(drone_spawn_count):
                        j=len(drone_spawn_count)-1
                    if elem-drone_spawn_count[j]>0:
                        drone_rate.append(100*drone_spawn_count[j]/(elem-drone_spawn_count[j]))
                    else:
                        drone_rate.append(0)
                drone_rate=np.array(drone_rate)
                axs[1][1].plot(enemy_spawn_times, drone_rate)
                axs[1][1].set_ylim(0,20)
                axs[0][1].plot(drone_spawn_times/60, drone_spawn_count/np.max(drone_spawn_count), label='Drones', c='b')

            axs[0][1].plot(enemy_spawn_times/60, enemy_spawn_count/np.max(enemy_spawn_count), label='Enemies', c='r')

            axs[1][0].scatter(drone_spawn_times/60, drone_spawn_count, s=2, c='b')
            axs[0][0].plot(enemy_spawn_times/60, enemy_spawn_count, c='r')

            axs[0][1].legend()
            axs[0][0].set_xlabel('Time (min)')
            axs[1][0].set_xlabel('Time (min)')
            axs[0][1].set_xlabel('Time (min)')
            axs[1][1].set_xlabel('Time (min)')
            print(len(drone_spawn_count))

            proc_data = get_proc_data()
            st = start_time+self.global_time
            i=0
            for elem in proc_data:
                if elem > drone_spawn_times[0]+st and elem < drone_spawn_times[-1]+st:
                    axs[1][0].axvspan((elem-st)/60, (elem-st+self.max_proc_time)/60, alpha=0.5, color='green', label='_'*i+'Smeeta proc')
                    i+=1
            if i > 0: axs[1][0].legend()

            fig.tight_layout()
            # get loot
            spawn_times = [f+st for f in drone_spawn_times]
            loot = self.get_expected_loot(spawn_times)
            self.ui.average_drops_label.setText(str(int(loot)))

            self.ui.drone_spawns_label.setText(str(drone_count))
            self.ui.total_spawns_label.setText(str(en_count))
            mission_t = int(mission_end_time-st)
            self.ui.mission_time_label.setText(str(datetime.timedelta(seconds=mission_t)))
            if mission_t>0:
                self.ui.drone_kpm_label.setText('%.2f, %.2f'%(drone_count/(mission_t/60),drone_count/(mission_t/3600)))
                self.ui.kpm_label.setText('%.2f'%(en_count/(mission_t/60)))
            if en_count>0:
                self.ui.drone_rate_label.setText('%.2f%%'%(100*drone_count/(en_count-drone_count)))
            else: self.ui.drone_rate_label.setText('-')

            plt.show()

    # spawn times epoch
    def get_expected_loot(self, spawn_times):
        dcb,db,bdcb=1,1,1
        if self.ui.drop_chance_booster_checkbox.isChecked():
            dcb = 2
        if self.ui.drop_booster_checkbox.isChecked():
            db = 2
        if self.ui.bless_booster_checkbox.isChecked():
            bdcb = 1.25
        dsdcb = self.ui.dark_sector_booster_spinner.value()

        proc_data = get_proc_data()
        valid_proc_data = [ f for f in proc_data if f>spawn_times[0] and f<spawn_times[-1] ]
        print(len(spawn_times))

        proc_index = 0
        loot=0
        for spawn_time in spawn_times:
            proc_hit_count=0
            for proc_time in valid_proc_data:
                if spawn_time>proc_time and spawn_time<(proc_time+self.max_proc_time):
                    proc_hit_count+=1
            loot += 2**proc_hit_count*dcb*db*bdcb*dsdcb*0.06
        return loot
#1644367086.2767293
def isfloat(value):
  try:
    float(value)
    return True
  except ValueError:
    return False

def get_proc_data():
    with open('smeeta_history.csv', newline='') as f:
        reader = csv.reader(f)
        time_list= list(reader)
    return [float(val[0]) for val in time_list]


