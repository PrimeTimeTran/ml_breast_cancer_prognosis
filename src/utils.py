import os
import sys
import pickle
import logging
from datetime import datetime

base_dir = os.path.dirname(os.path.abspath(__file__))
save_dir = os.path.join(base_dir, '../tmp/output')
plots_dir = os.path.join(base_dir, '../tmp/plots')
logs_dir = os.path.join(base_dir, '../tmp/logs')

def plot_file_name(model_type, train_scope, set_type):
    return os.path.join(plots_dir, f'{model_type}-{train_scope}-{set_type}-confusion-matrix.png')

def plot_graph_name(model_type, train_scope):
    return os.path.join(plots_dir, f'{model_type}-{train_scope}-graph.png')

def load_pickle(model):
    try:
        with open(f'tmp/models/{model}_DBT.pickle', 'rb') as f:
            loaded_model = pickle.load(f)
        return loaded_model
    except FileNotFoundError:
        print(f"Error: The file 'tmp/models/{model}_DBT.pickle' was not found.")
        return None
    except Exception as e:
        print(f"Error loading the pickle file: {str(e)}")
        return None


def setup_save_directory():
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)

def image_file_name(type, patient_id, label, predicted_label):
    return os.path.join(
        save_dir, f'{type}-{patient_id}-labeled-{label}-labeled-by-model-{predicted_label}.png')

def setup_logger(name):
    current_datetime = datetime.now().strftime('%m-%d-%y_%H-%M-%S')
    log_filename = f"{current_datetime}_{name}-summary.log"
    log_path = os.path.join(logs_dir, log_filename)

    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    console_handler = logging.StreamHandler(sys.stdout)
    file_handler = logging.FileHandler(log_path)
    console_handler.setLevel(logging.DEBUG)
    file_handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    return logger

def get_data_file(set_type):
    return os.path.join(base_dir, f'../tmp/{set_type}/BCS-DBT-file-paths-{set_type}-v2.csv')


def get_files(set_type):
    return [
        os.path.join(
            base_dir, f'../tmp/{set_type}/BCS-DBT-file-paths-{set_type}-v2.csv'),
        os.path.join(
            base_dir, f'../tmp/{set_type}/BCS-DBT-labels-{set_type}-v2.csv'),
        os.path.join(
            base_dir, f'../tmp/{set_type}/BCS-DBT-boxes-{set_type}-v2.csv')
    ]
