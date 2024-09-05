import subprocess
import sys
import os


if __name__ == "__main__":
    # get path to this file (equal to the project's path)
    script_path = os.path.abspath(__file__)
    project_dir = os.path.dirname(script_path)
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-e", f"{project_dir}"])
