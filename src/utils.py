import os
import shutil
import numpy as np

base_dir = os.path.dirname(os.path.abspath(__file__))
save_dir = os.path.join(base_dir, '../tmp/output')
logs_dir = os.path.join(base_dir, '../tmp/logs')

def get_file_name(type, model_type):
    return os.path.join(save_dir, f'../matrices/{model_type}-confusion-matrix-for-{type}-data.png')

def setup_save_directory():
    if os.path.exists(save_dir):
        shutil.rmtree(save_dir)
    os.makedirs(save_dir, exist_ok=True)

def image_file_name(type, idx, value):
    return os.path.join(
        save_dir, f'{type}-{idx}-original-{value}-predict-{value}.png')

def create_log_file(name):
    return open(f'{logs_dir}/{name}', "w")
