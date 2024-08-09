import argparse

import numpy as np

import os

if __name__ == "__main__":
    split_file = 'data/waymo_splits/static32.txt'
    split_file = open(split_file, "r").readlines()[1:]
    scene_ids_list = [int(line.strip().split(",")[0]) for line in split_file]
    os.makedirs('data/waymo/processed/static/training', exist_ok=True)
    for scene_id in scene_ids_list:
        scene_str = f'{scene_id:03d}'
        scene_path = os.path.abspath(f'data/waymo/processed/training/{scene_str}')
        os.system(f'ln -s {scene_path} data/waymo/processed/static/training/{scene_str}')
