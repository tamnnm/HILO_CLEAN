import argparse
import subprocess
import os
from pathlib import Path

# Source the shell script and capture the environment variables
def source_shell(path="/home/tamnnm/",file_name="load_env_vars.sh"):
    command = f'source {path}{file_name} && env'
    proc = subprocess.Popen(['bash', '-c', command], stdout=subprocess.PIPE, universal_newlines=True)
    env_vars = {}
    for line in proc.stdout:
        key, _, value = line.partition("=")
        env_vars[key.strip()] = value.strip()
    proc.communicate()
    return env_vars
