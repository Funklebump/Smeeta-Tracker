import git
import os
import subprocess

repo_dir = os.path.join(os.getcwd())
repo = git.Repo.init(repo_dir)
repo.head.reset('HEAD~1', index=True, working_tree=True)
git.cmd.Git().pull('https://github.com/A-DYB/smeeta-tracker-2' , 'main')

subprocess.Popen(['python', './main.py'])