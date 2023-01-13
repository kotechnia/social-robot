from __future__ import division, print_function

import os
import torch
import torchvision
import json
import datetime
import time
import numpy as np
import pandas as pd
import pickle as pkl
import sys

from torchvision import transforms
from torch import nn

from video_backbone.TSP.extract_features.eval_video_dataset import EvalVideoDataset
from video_backbone.TSP.common import utils
from video_backbone.TSP.common import transforms as T
from video_backbone.TSP.models.model import Model

import os
from multiprocessing import Pool
import multiprocessing as mp
import subprocess
from glob import glob
from joblib import Parallel, delayed
from torchvision.io import read_video_timestamps
from tqdm import tqdm


def get_video_stats(filename):
    pts, video_fps = read_video_timestamps(filename=filename, pts_unit='sec')
    if video_fps:
        stats = {'filename': filename,
                 'video-duration': len(pts)/video_fps,
                 'fps': video_fps,
                 'video-frames': len(pts)}
    else:
        stats = {'filename': filename,
                 'video-duration': None,
                 'fps': None,
                 'video-frames': None}
        print(f'WARNING: {filename} has an issue. video_fps = {video_fps}, len(pts) = {len(pts)}.')
    return stats


def GPU_free_check():
    try:
        import nvidia_smi
    except ModuleNotFoundError as e:
        print("you must install 'pip install nvidia-ml-py3'")
        raise e
    
    nvidia_smi.nvmlInit()
    deviceCount = nvidia_smi.nvmlDeviceGetCount()
    device_info = {}
    for i in range(deviceCount):
        handle = nvidia_smi.nvmlDeviceGetHandleByIndex(i)
        info = nvidia_smi.nvmlDeviceGetMemoryInfo(handle)
        device_info[str(i)] = (info.free) / (1024 * 1024)
    nvidia_smi.nvmlShutdown()
    return device_info

def GPU_available(volume, shuffle=False):
    
    device_info = GPU_free_check()
    device_indices = list(device_info.keys())
    
    if shuffle:
        import random
        random.shuffle(device_indices)
    
    for device_id in device_indices:
        if device_info[device_id] > volume:
            return device_id
    
    return None


def f(x):
    command='python video_backbone/TSP/extract_features/extract_features.py --data-path . --metadata-csv-filename {csv_path} --released-checkpoint "r2plus1d_34-tsp_on_activitynet" --stride 16 --shard-id 0 --num-shards 1 --device cuda:{gpu_num} --output-dir {output_dir} --workers 1'
    command=command.format(gpu_num=x['gpu_num'], csv_path=x['csv_path'], output_dir=x['output_dir'])
    print(command)

    try:
        output = subprocess.check_output(command, shell=True, encoding='utf-8')
    except (subprocess.CalledProcessError) as e:
        output = e
    #print(output, x, os.getpid())
    return x


def main(args):
    video_root = args.video_root
    feature_root = args.feature_root
    process_core = int(args.num_workers)

    memory = GPU_free_check()
    process_num = 0
    for key in memory.keys():
        # 8654 tsn memory
        local_process_num = memory[key] // 9000
        memory[key] = local_process_num
        process_num += int(local_process_num)

    if process_core > mp.cpu_count():
        process_core = mp.cpu_count()

    filenames = glob(os.path.join(video_root, '**/*.mp4'), recursive=True)

    all_stats = Parallel(n_jobs=process_core)(
        delayed(get_video_stats)(
            filename=filename,
        ) for filename in tqdm(filenames))
    df = pd.DataFrame(all_stats)

    #df.to_csv('metadata.csv', index=False)
    #df = pd.read_csv('metadata.csv')

    #process_num=1
    length = df.shape[0]
    indices = np.linspace(0, length, process_num+1)
    process_params=[]

    for i in range(len(indices)-1):
        csv_path = f'metadata_{i}.csv'
        df.loc[indices[i]:indices[i+1]].reset_index(drop=True).to_csv(csv_path, index=False)

        for key in memory.keys():
            if memory[key] != 0:
                memory[key] -= 1
                gpu_num=int(key)
                break
    
        process_params.append({"gpu_num":gpu_num, "csv_path":csv_path, "output_dir":feature_root})

    with Pool(process_num) as p:
        print(p.map(f, process_params))

    for i in range(len(indices)-1):
        csv_path = f'metadata_{i}.csv'
        os.remove(csv_path)
        

if __name__ == '__main__':
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.allow_tf32 = True

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--video_root', default='../data/videos', type=str)
    parser.add_argument('--feature_root', default='../data/features')
    parser.add_argument('--num_workers', default=20, type=int)
    args = parser.parse_args()

    main(args)
