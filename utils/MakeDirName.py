import os
import time


def get_dir_rank_name(path):
    all_folders = os.listdir(path)
    if len(all_folders) == 0:
        return path + 'PPO_1'
    all_folders.sort()
    latest = all_folders[-1].replace('PPO_', '')
    return path + 'PPO_' + str(int(latest) + 1)


def get_dir_data_name(path):
    time_name = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime())
    return path + time_name

