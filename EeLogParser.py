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

    def plot_logs(self):
        fig, axs = plt.subplots(2,2)
        df = self.parse_arbitration_logs()
        if df is None:
            print(f'No mission data')
            return

        sns.lineplot(data=df, x='mission_time_minutes', y='drones_per_hour', ax=axs[1][1], errorbar=None)
        sns.lineplot(data=df, x='mission_time_minutes', y='drones_per_enemy', ax=axs[0][1], errorbar=None)
        sns.lineplot(data=df, x='mission_time_minutes', y='enemy_count', ax=axs[0][0], errorbar=None)
        sns.lineplot(data=df, x='mission_time_minutes', y='drone_count', ax=axs[1][0], errorbar=None)

        axs[0][0].set_title('Enemy spawns')
        axs[1][0].set_title('Drone spawns')
        axs[0][1].set_title('Drones per enemy')
        axs[1][1].set_title('Drones per hour')

        df.sort_values(by='mission_time_minutes', inplace=True)
        vitus_chance = self.get_vitus_essence_chance()
        df['boost'] = np.array([vitus_chance]*len(df.index))
        df['drone_count_diff'] = np.diff(df.drone_count.to_numpy(), prepend=0)

        df_s = pd.read_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'smeeta_history.csv'))
        smeeta_proc_timestamps_unix_s = df_s['smeeta_proc_unix_s'].to_numpy()

        mission_start_time_unix_s = df.timestamp_unix_s.min()
        mission_end_time_unix_s = df.timestamp_unix_s.max()
        mission_smeeta_proc_timestamps_unix_s = smeeta_proc_timestamps_unix_s[np.where((smeeta_proc_timestamps_unix_s>mission_start_time_unix_s) & (smeeta_proc_timestamps_unix_s < mission_end_time_unix_s))]
        for smeeta_proc_timestamp in mission_smeeta_proc_timestamps_unix_s:
            axs[1][0].axvspan((smeeta_proc_timestamp - mission_start_time_unix_s)/60, (smeeta_proc_timestamp - mission_start_time_unix_s + self.window.affinity_proc_duration)/60, alpha=0.5, color='green')
            df.loc[ (df.timestamp_unix_s > smeeta_proc_timestamp) & (df.timestamp_unix_s < smeeta_proc_timestamp + self.window.affinity_proc_duration), 'boost'] *= 2
        axs[1][0].legend(['Smeeta proc'])

        total_vitus_essence = np.sum( np.multiply( df.boost.to_numpy(), df.drone_count_diff.to_numpy() ) )

        # get last mission
        mission_info_str = self.get_node_info_string(self.recently_played_mission)

        plt.suptitle(f'{mission_info_str}\nDrones: {df.drone_count.max()}\nAvg VE Drops: {total_vitus_essence:.0f}; Avg Boost: {total_vitus_essence/(max(1,df.drone_count.max())*0.06):.2f}')
        fig.tight_layout()
        
        plt.show()

    def parse_arbitration_logs(self):
        if not os.path.isfile(self.ee_log_path):
            return
        
        data = []
        drone_count = 0
        enemy_count = 0
        latest_log_time_s = None
        mission_end_time_s = None

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
                    data.append({'log_time_s':log_time_s, 'drone_count':drone_count, 'enemy_count':enemy_count})
                    break
                elif are_elems_in_line(constants.MISSION_END_TEXT, line):
                    mission_end_time_s = self.game_start_time_unix_s + log_time_s
                elif are_elems_in_line(constants.DRONE_AGENT_CREATED_TEXT, line):
                    drone_count+=1
                    data.append({'log_time_s':log_time_s, 'drone_count':drone_count, 'enemy_count':enemy_count})
                elif are_elems_in_line(constants.AGENT_CREATED_TEXT, line) and not are_elems_in_line(constants.INVALID_AGENT_CREATED_TEXT, line):
                    enemy_count+=1
                    data.append({'log_time_s':log_time_s, 'drone_count':drone_count, 'enemy_count':enemy_count})
                elif are_elems_in_line(constants.DRONE_AGENT_DESTROYED_TEXT, line):
                    drone_count-=1
                    data.append({'log_time_s':log_time_s, 'drone_count':drone_count, 'enemy_count':enemy_count})

        if mission_end_time_s is None:
            mission_end_time_s = latest_log_time_s

        if len(data) == 0:
            return

        df = pd.DataFrame(data)
        df.drone_count = np.abs( df.drone_count.to_numpy() - df.drone_count.max() )
        df.enemy_count = np.abs( df.enemy_count.to_numpy() - df.enemy_count.max() )
        df['timestamp_unix_s'] = df.log_time_s.to_numpy() + self.game_start_time_unix_s
        df['mission_time_s'] = df.log_time_s.to_numpy() - df.log_time_s.min()
        df['mission_time_minutes'] = df.mission_time_s.to_numpy()/60
        df['drones_per_hour'] = np.divide(df.drone_count.to_numpy()*3600, np.clip(df.mission_time_s.to_numpy(), 1, None))
        df['drones_per_enemy'] = np.divide(df.drone_count.to_numpy(), np.clip(df.enemy_count.to_numpy(), 1, None))
        df['enemies_per_hour'] = np.divide(df.enemy_count.to_numpy()*3600, np.clip(df.mission_time_s.to_numpy(), 1, None))

        #last_mission_node = self.get_recent_node_name("Script [Info]: ThemedSquadOverlay.lua: Host loading {\"name\":")
        if self.recently_played_mission is not None:
            with open(os.path.join(self.dirname,"solNodes.json")) as f:
                map_info = json.load(f)
            node_info = map_info.get(self.recently_played_mission)
            if node_info:
                PB = node_info.get("personal_best_dph", 0)
                if df.drones_per_hour.iloc[-1] > PB:
                    map_info[self.recently_played_mission]["personal_best_dph"] = df.drones_per_hour.iloc[-1]
                    # save 
                    with open(os.path.join(self.dirname,"solNodes.json"), "r+") as fp:
                        fp.truncate(0)
                        json.dump(map_info , fp) 

        return df
    
    def get_vitus_essence_chance(self):
        return self.window.window_data.drop_chance_booster * self.window.window_data.drop_booster * self.window.window_data.drop_booster2 * \
                    self.window.window_data.bless_booster * self.window.window_data.dark_sector_booster * 0.06

    def get_node_info_string(self, node):
        mission_info_str=''
        if not node:
            return mission_info_str
        with open(os.path.join(self.dirname,"solNodes.json"), encoding='utf-8') as f:
            map_info = json.load(f)
        node_info = map_info.get(node)
        if node_info:
            mission_info_str = f'{node_info["value"]} - {node_info["enemy"]} {node_info["type"]}'
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
    