import os
import shutil
import numpy as np

base_dir = os.path.dirname(os.path.abspath(__file__))
save_dir = os.path.join(base_dir, '../tmp/output')
plots_dir = os.path.join(base_dir, '../tmp/plots')
logs_dir = os.path.join(base_dir, '../tmp/logs')

def plot_file_name(type, model_type):
    return os.path.join(save_dir, f'../plots/{model_type}-confusion-matrix-for-{type}-data.png')

def setup_save_directory():
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)

def image_file_name(type, idx, value):
    return os.path.join(
        save_dir, f'{type}-{idx}-original-{value}-predict-{value}.png')

def create_log_file(name):
    log_path = os.path.join(logs_dir, name)
    if os.path.exists(log_path):
        prev_log_path = log_path.replace('.log', '-prev.log')
        if os.path.exists(prev_log_path):
            os.remove(prev_log_path)
        shutil.move(log_path, prev_log_path)
    return open(log_path, "w")
