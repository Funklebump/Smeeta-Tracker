# smeeta-tracker-2
Track smeeta affinity procs with an overlay and sound notifications. 

Contains additional tools for tracking Arbitration Drones in arbitration runs.

# Warnings
This application is not endorsed by Digital Extremes and is fan-made. It processes pictures of your screen and reads your ee.log file (which is meant to be user-accessible). However, it's important to use your own judgement and use it at your own risk. 

According to section 2.f of https://www.warframe.com/EULA, you agree that you will not under any circumstance "use... unauthorized third-party software, tools or content designed to modify the ...Game experience". By using any kind of macros, overlays, or any third-party tool (including this one), you are breaking this clause to my understanding. You should read the EULA yourself, read the code to see what it is doing and come to your own conclusion if you want to use this tool. Refer to this PSA from Digital Extremes about third-party software to get an idea of their stance: https://forums.warframe.com/topic/1320042-third-party-software-and-you/.

# How to use
This program only works on Windows, and was only tested on Windows 10.

## Running the executable
1. Download the latest release from https://github.com/A-DYB/smeeta-tracker-2/releases
2. If you want to keep your charm history and mission history, tranfer your "user_ExportRegions.json" and "charm_history.csv" files to the new directory
3. Double click "Smeeta Tracker.exe"

Note: If the directory is a mess and you want a clean view in File Explorer, make sure under "View" tab, "Hidden Items" is unchecked

## Running as a Python Script
1. Download the project files. (Or use "git clone https://github.com/A-DYB/smeeta-tracker-2.git")
2. Make sure you have python 3.11.2 installed. https://www.python.org/downloads/release/python-3112/
3. Open the project folder in Windows File Explorer
4. Open command prompt by typing "cmd" in the Windows file explorer's navigation bar.
5. Make sure your python version is correct by typing "python --version"
5. Type "python -m venv ./venv" to create a virtual environment where we will install all the packages
6. Once created you can double click the file "open_venv.bat" to open a command prompt with the virtual environment activated. Double click this file and a command prompt window should appear.
7. Type "py -m pip install -r requirements.txt" in the command prompt and make sure it completes successfully 
8. Run the program "main.py" by typing "python main.py" in cmd, or right click, "Open With", "Python"

# Features
- [x] Track Smeeta's affinity procs using an in-game overlay, and optionally using sounds
- [x] Hosts can display information in the overlay related to Arbitration drone spawns
- [x] Hosts can generate a mission summary displaying mission statistics
- [x] Display current Arbitration mission in overlay and your Personal Best drone kills per hour for that node

# Smeeta detector tips
- The more visible your in-game text, the better the program will work. To improve accuracy change your UI colors for text and buffs to uncommon, highly saturated colors
- Keep brightness and contrast settings at 50% (modifying these can change the contrast between the text and background making them less detectable)

# Arbitration Logger tips
- Only works for the Host because only the host's log file is updated with mission info. You are host if there is a 1 next to your name in mission.
