import os
import shutil
import pickle
import numpy as np
import struct

base_dir = os.path.dirname(os.path.abspath(__file__))
save_dir = os.path.join(base_dir, '../../tmp/output')
logs_dir = os.path.join(base_dir, '../../tmp/logs')
data_dir = os.path.join(base_dir, '../../tmp/dataset')

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

def create_pickle(clf, model_type):
    with open(f'tmp/models/{model_type}_DBT.pickle', 'wb') as f:
        pickle.dump(clf, f)
    pickle_in = open(f'tmp/models/{model_type}_DBT.pickle', 'rb')
    clf = pickle.load(pickle_in)
    return clf

def load_mnist_images(filename):
    with open(filename, 'rb') as f:
        magic, num, rows, cols = struct.unpack(">IIII", f.read(16))
        images = np.fromfile(f, dtype=np.uint8).reshape(num, rows, cols)
        return images

def load_mnist_labels(filename):
    with open(filename, 'rb') as f:
        magic, num = struct.unpack(">II", f.read(8))
        labels = np.fromfile(f, dtype=np.uint8)
        return labels
