import psutil
import time
import argparse

def monitor_process(pid):
    while True:
        try:
            process = psutil.Process(pid)
            memory_usage_gb = process.memory_info().rss / (1024 ** 3)  # convert to GB
            print('Memory usage: {:.2f} GB'.format(memory_usage_gb))
        except psutil.NoSuchProcess:
            print("Process no longer exists")
            break
        time.sleep(1)

parser = argparse.ArgumentParser(description='Monitor memory usage of a process.')
parser.add_argument('pid', type=int, help='The PID of the process to monitor.')

args = parser.parse_args()

monitor_process(args.pid)
