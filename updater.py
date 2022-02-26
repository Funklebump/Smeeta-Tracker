import git
import os
import subprocess

repo_dir = os.path.join(os.getcwd())
repo = git.Repo.init(repo_dir)
repo.git.add('*')
#repo.git.stash()
#repo.git.reset('--hard')
git.cmd.Git().pull('https://github.com/A-DYB/smeeta-tracker-2' , 'main')

subprocess.Popen(['python', './main.py'])