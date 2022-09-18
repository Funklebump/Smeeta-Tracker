# smeeta-tracker-2
Track smeeta affinity procs with an overlay and sound notifications. 

Contains additional tools for tracking Arbitration Drones in arbitration runs.

# Warnings
This application is fan made and not endorsed by Digital Extremes. This does nothing that is against TOS (all it does is process pictures of your screen and reads your log file), but use your own judgement and use at your own risk.

This program only works on Windows, and was only tested on Windows 10.

# How to use
## Python
1. Download the project files. (Or use "git clone https://github.com/A-DYB/smeeta-tracker-2.git")
2. Make sure you have python installed. https://www.python.org/downloads/
3. (Optional) Make sure you have git installed. This will make updates easy, all you have to do is push a button in the UI. https://git-scm.com/downloads
4. Open the project folder in Windows File Explorer
5. Open command prompt by typing "cmd" in the Windows file explorer's navigation bar
4. Type "py -m pip install -r requirements.txt" in command prompt
5. Run the program "main.py" by typing "python main.py" in cmd, or right click, "Open With", "Python"

# Features
- [x] Track Smeeta's affinity procs using an in-game overlay, and optionally using sounds
- [x] Hosts can display information in the overlay related to Arbitration drone spawns
- [x] Hosts can generate a mission summary displaying mission statistics
- [x] Display current Arbitration mission in overlay and your Personal Best drone kills per hour for that node

# Smeeta detector tips
- The more visible your in-game text, the better the program will work. To improve accuracy change your UI colors for text and buffs to uncommon, highly saturated colors
- Keep brightness and contrast settings at 50% (modifying these can change the contrast between the text and background making them less detectable)

# Arbitration Logger tips
- Only works for Host because only the host's log file is updated with mission info

# Acknowledgments
Thanks to https://github.com/WFCD/warframe-worldstate-data for solar node information file
