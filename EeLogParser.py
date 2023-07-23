import time
import io
from lz.reversal import reverse
import os
from win32com.shell import shell, shellcon
import calendar
import datetime
import matplotlib.pyplot as plt
import numpy as np
import re
import json
import pandas as pd
import seaborn as sns
import constants

class EeLogParser:
    def __init__(self, main_window):
        self.window = main_window
        self.dirname = os.path.dirname(os.path.abspath(__file__))
        self.ee_log_path = os.path.join(shell.SHGetFolderPath(0, shellcon.CSIDL_LOCAL_APPDATA, None, 0), 'Warframe\ee.log')

        self.user_ExportRegions = None
        if os.path.isfile(os.path.join(self.dirname,"user_ExportRegions.json")):
            with open(os.path.join(self.dirname,"user_ExportRegions.json"), encoding='utf-8') as f:
                self.user_ExportRegions = json.load(f)

        self.current_arbitration = self.get_recent_node_name("Script [Info]: Background.lua: EliteAlertMission at ")
        #TODO change usage of current_arbitration
        self.recently_played_mission = self.get_recent_node_name("Script [Info]: ThemedSquadOverlay.lua: Host loading {\"name\":")

        self.previous_log_timestamp_s=0
        self.latest_log_timestamp_s=0

        self.game_start_time_unix_s = self.sync_time()

        self.in_mission=False
        self.mission_start_timestamp_s = 0
        self.mission_start_timestamp_unix_s = 0
        self.mission_end_timestamp_s = 0
        self.mission_duration_s=0
        self.drone_spawns = 0
        self.enemy_spawns = 0
        self.drones_per_hour = 0

    def reset(self):
        self.user_ExportRegions = None
        if os.path.isfile(os.path.join(self.dirname,"user_ExportRegions.json")):
            with open(os.path.join(self.dirname,"user_ExportRegions.json"), encoding='utf-8') as f:
                self.user_ExportRegions = json.load(f)

        self.current_arbitration = self.get_recent_node_name("Script [Info]: Background.lua: EliteAlertMission at ")
        self.recently_played_mission = self.get_recent_node_name("Script [Info]: ThemedSquadOverlay.lua: Host loading {\"name\":")

        self.previous_log_timestamp_s=0
        self.latest_log_timestamp_s=0

        self.game_start_time_unix_s = self.sync_time()

        self.in_mission=False
        self.mission_start_timestamp_s = 0
        self.mission_start_timestamp_unix_s = 0
        self.mission_end_timestamp_s = 0
        self.mission_duration_s=0
        self.drone_spawns = 0
        self.enemy_spawns = 0
        self.drones_per_hour = 0 

    def sync_time(self):
        if not os.path.isfile(self.ee_log_path):
            return
        with open(self.ee_log_path, encoding="utf8", errors='replace') as log_file:
            for line in log_file:
                # find time that game was started
                if "Sys [Diag]: Current time:" in line:
                    res = line.split("[")[2]
                    res = res.split()
                    s = res[3]+"-"+res[2]+"-"+res[5].split("]")[0]+":"+res[4]
                    timestruct = time.strptime(s, "%d-%b-%Y:%H:%M:%S")
                    log_timestamp_s = self.get_log_timestamp_s(line)
                    if log_timestamp_s:
                        return calendar.timegm(timestruct) - log_timestamp_s
                    else:
                        return calendar.timegm(timestruct)
            raise Exception("Could not find log file time reference")

    def parse_latest_logs(self):
        if not os.path.isfile(self.ee_log_path):
            return
        found_arb=False
        found_end = False
        found_start = False
        with open(self.ee_log_path, encoding="utf8", errors="replace") as log_file:
            for line_index, line in enumerate(reverse(log_file, batch_size=io.DEFAULT_BUFFER_SIZE)):
                # skip first line, it may still be being written to
                if line_index == 0:
                    continue

                log_timestamp_s = self.get_log_timestamp_s(line)
                if log_timestamp_s is None:
                    continue
                if line_index == 1:
                    self.latest_log_timestamp_s = log_timestamp_s

                # Check if we are up to date on latest logs
                if log_timestamp_s <= self.previous_log_timestamp_s:
                    break

                # check if mission has ended
                if are_elems_in_line(constants.MISSION_END_TEXT, line) and not found_end and not found_start:
                    found_end = True
                    print(f'[{log_timestamp_s}]: Mission has ended')
                    self.mission_end_timestamp_s = log_timestamp_s
                    if self.drone_spawns > 0:
                        # save logs to file
                        self.parse_arbitration_logs()
                    self.in_mission = False
                    self.window.monitor.screen_scanner.reset()
                    self.drone_spawns = 0
                    self.enemy_spawns = 0
                    self.drones_per_hour = 0
                    
                # Check if drone has spawned
                elif are_elems_in_line(constants.DRONE_AGENT_CREATED_TEXT, line):
                    self.drone_spawns += 1
                # Check if enemy has spawned
                elif are_elems_in_line(constants.AGENT_CREATED_TEXT, line) and not are_elems_in_line(constants.INVALID_AGENT_CREATED_TEXT, line):
                    self.enemy_spawns+=1
                elif are_elems_in_line(constants.DRONE_AGENT_DESTROYED_TEXT, line):
                    self.drone_spawns-=1
                # Check for mission start
                elif are_elems_in_line(constants.MISSION_START_TEXT, line) and not found_start:
                    found_start = True
                    if not found_end:
                        print(f'[{log_timestamp_s}]: Mission has started')
                        self.in_mission = True
                        
                    self.mission_start_timestamp_s = log_timestamp_s
                    self.mission_start_timestamp_unix_s = log_timestamp_s + self.game_start_time_unix_s
                    if self.drone_spawns > 0 and found_end:
                        # save logs to file
                        self.parse_arbitration_logs()
                    break
                # check for mission start name
                elif are_elems_in_line(constants.MISSION_NAME_TEXT, line):
                    recently_played_mission = self.parse_node_string(line)
                    if recently_played_mission:
                        self.recently_played_mission = recently_played_mission
                # check for new arbitration name
                elif are_elems_in_line(constants.NEW_ARBITRATION_NAME_TEXT, line) and not found_arb:
                    found_arb = True
                    current_arbitration = self.parse_node_string(line)
                    if current_arbitration:
                        self.current_arbitration = current_arbitration

        self.previous_log_timestamp_s = self.latest_log_timestamp_s
        self.mission_duration_s = self.latest_log_timestamp_s - self.mission_start_timestamp_s  if self.in_mission else self.mission_end_timestamp_s - self.mission_start_timestamp_s
        self.drones_per_hour = 3600*self.drone_spawns/(max(1, self.mission_duration_s))

    def get_log_timestamp_s(self, line):
        line_split = line.split(" ")
        if len(line_split) == 0:
            return
        timestamp_str = line_split[0]
        match_object = re.search(r'\d*[.]\d{3}', timestamp_str)
        if match_object is None:
            return
        if not isfloat(timestamp_str):
            return
        return float(timestamp_str)

    def get_recent_node_name(self, search_condition):
        if not os.path.isfile(self.ee_log_path):
            return
        with open(self.ee_log_path, encoding="utf8", errors='replace') as log_file:
            for line in reverse(log_file, batch_size=io.DEFAULT_BUFFER_SIZE):
                if search_condition in line:
                    return self.parse_node_string(line)

    def parse_node_string(self, line):
        for node_type in [r'SolNode', 'ClanNode', 'SettlementNode']:
            match = re.search(node_type + r'[\d]+', line)
            if match:
                break
        if not match:
            return
        return match.group(0)

    def parse_arbitration_logs(self):
        if not os.path.isfile(self.ee_log_path):
            return
        self.sync_time()
        
        data = []
        drone_count = 0
        enemy_count = 0
        latest_log_time_s = None
        mission_end_time_unix_s = None
        mission_start_time_unix_s = None

        with open(self.ee_log_path, encoding="utf8", errors='replace') as log_file:
            line_stitch = ''
            for line_index, line in enumerate(reverse(log_file, batch_size=io.DEFAULT_BUFFER_SIZE)):
                log_time_s = self.get_log_timestamp_s(line)
                if log_time_s is None:
                    line_stitch = line + line_stitch
                    continue
                if line_stitch != '':
                    line = line + line_stitch
                    line_stitch = ''
                line = line.replace('\n', '').replace('\r', '')

                if latest_log_time_s is None:
                    latest_log_time_s = log_time_s
     
                if are_elems_in_line(constants.MISSION_START_TEXT, line):
                    mission_start_time_unix_s = self.game_start_time_unix_s + log_time_s
                    data.append({'log_time_s':log_time_s, 'drone_count':drone_count, 'enemy_count':enemy_count})
                    break
                elif are_elems_in_line(constants.MISSION_END_TEXT, line):
                    mission_end_time_unix_s = self.game_start_time_unix_s + log_time_s
                elif are_elems_in_line(constants.DRONE_AGENT_CREATED_TEXT, line):
                    drone_count+=1
                    data.append({'log_time_s':log_time_s, 'drone_count':drone_count, 'enemy_count':enemy_count})
                elif are_elems_in_line(constants.AGENT_CREATED_TEXT, line) and not are_elems_in_line(constants.INVALID_AGENT_CREATED_TEXT, line):
                    enemy_count+=1
                    data.append({'log_time_s':log_time_s, 'drone_count':drone_count, 'enemy_count':enemy_count})
                elif are_elems_in_line(constants.DRONE_AGENT_DESTROYED_TEXT, line):
                    drone_count-=1
                    data.append({'log_time_s':log_time_s, 'drone_count':drone_count, 'enemy_count':enemy_count})

        if len(data) == 0:
            return
        df = pd.DataFrame(data)

        if mission_start_time_unix_s is not None and mission_end_time_unix_s is not None:
            self.save_mission_data(df, mission_start_time_unix_s, mission_end_time_unix_s)

        if mission_end_time_unix_s is None:
            mission_end_time_unix_s = self.game_start_time_unix_s + latest_log_time_s 

        if mission_start_time_unix_s is None:
            mission_start_time_unix_s = self.game_start_time_unix_s + log_time_s 

        
        df.drone_count = np.abs( df.drone_count.to_numpy() - df.drone_count.max() )
        df.enemy_count = np.abs( df.enemy_count.to_numpy() - df.enemy_count.max() )
        df['timestamp_unix_s'] = df.log_time_s.to_numpy() + self.game_start_time_unix_s
        df['mission_time_s'] = df.log_time_s.to_numpy() - df.log_time_s.min()
        df['mission_time_minutes'] = df.mission_time_s.to_numpy()/60
        df['drones_per_hour'] = np.divide(df.drone_count.to_numpy()*3600, np.clip(df.mission_time_s.to_numpy(), 1, None))
        df['drones_per_enemy'] = np.divide(df.drone_count.to_numpy(), np.clip(df.enemy_count.to_numpy(), 1, None))
        df['enemies_per_hour'] = np.divide(df.enemy_count.to_numpy()*3600, np.clip(df.mission_time_s.to_numpy(), 1, None))

        return df
    
    def save_mission_data(self, df:pd.DataFrame, mission_start_time_unix_s, mission_end_time_unix_s):
        if not os.path.isfile(os.path.join(self.dirname,"user_ExportRegions.json")):
            return
        with open(os.path.join(self.dirname,"user_ExportRegions.json")) as f:
            data = json.load(f)
        map_list = data.get('ExportRegions')
        if map_list is None:
            return
        
        mission_data = next((item for item in map_list if item.get("uniqueName", "") == self.recently_played_mission), None)
        if mission_data is None:
            return
        
        previous_run_list = mission_data.get('previous_run_list')
        new_mission_data = {"mission_start_time_unix_s":float(mission_start_time_unix_s), "mission_end_time_unix_s":float(mission_end_time_unix_s), 
                            "arbitration_drone_spawn_count":int(df["drone_count"].max()), "enemy_spawn_count":int(df["enemy_count"].max())}

        if previous_run_list is None:
            mission_data["previous_run_list"] = [new_mission_data]
        else:
            # verify that this mission was not already saved
            mission = next((item for item in previous_run_list if int(item.get("mission_start_time_unix_s", -1)) == int(mission_start_time_unix_s)), None)
            if mission is not None:
                return
            mission_data["previous_run_list"] = mission_data["previous_run_list"].append(new_mission_data)

        with open(os.path.join(self.dirname,"user_ExportRegions.json"), "r+") as fp:
            fp.truncate(0)
            json.dump(data , fp) 
    
    def get_vitus_essence_chance(self):
        return self.window.window_data.drop_chance_booster * self.window.window_data.drop_booster * self.window.window_data.drop_booster2 * \
                    self.window.window_data.bless_booster * self.window.window_data.dark_sector_booster * 0.06

    def get_node_info_string(self, node):
        mission_info_str=''
        if node is None:
            return mission_info_str
        
        if self.user_ExportRegions is None:
            return
        
        map_list = self.user_ExportRegions.get('ExportRegions')
        if map_list is None:
            return
        
        mission_data = next((item for item in map_list if item.get("uniqueName", "") == node), None)
        if mission_data is None:
            return
        
        name = mission_data.get("name", "")
        systemName = mission_data.get("systemName", "")
        factionName = constants.FACTIONINDEX_NAME.get(mission_data.get("factionIndex", -1), "")
        missionName = constants.MISSIONINDEX_NAME.get(mission_data.get("missionIndex", -1), "")

        mission_info_str = f'{name} ({systemName}), {factionName} {missionName}'
        return mission_info_str

def are_elems_in_line(elems:list, line:str):
    for elem in elems:
        if elem in line:
            return True
    return False

def isfloat(value):
    try:
        float(value)
        return True
    except ValueError:
        return False
    