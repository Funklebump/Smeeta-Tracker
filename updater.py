import git
import os

repo_dir = os.path.join(os.getcwd())
repo = git.Repo.init(repo_dir)
git.cmd.Git().pull('https://github.com/A-DYB/smeeta-tracker-2' , 'main')