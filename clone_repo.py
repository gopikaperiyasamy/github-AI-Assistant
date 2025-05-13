from git import Repo
import os

def clone_repo(git_url, clone_dir="cloned_repo"):
    if os.path.exists(clone_dir):
        print(f"Repo already cloned at {clone_dir}")
    else:
        Repo.clone_from(git_url, clone_dir)
        print(f"Cloned repo to {clone_dir}")
    return clone_dir
