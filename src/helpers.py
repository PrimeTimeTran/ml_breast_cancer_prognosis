import os
import pydicom
import numpy as np

script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(os.getcwd(), os.pardir))


def get_image_laterality(pixel_array: np.ndarray) -> str:
    left_edge = np.sum(pixel_array[:, 0])
    right_edge = np.sum(pixel_array[:, -1])
    return "R" if left_edge < right_edge else "L"


def get_window_center(ds: pydicom.dataset.FileDataset) -> np.float32:
    try:
        return np.float32(ds[0x5200, 0x9229][0][0x0028, 0x9132][0][0x0028, 0x1050].value)
    except KeyError:
        raise ValueError(
            "Window Center information not found in DICOM metadata.")


def get_window_width(ds: pydicom.dataset.FileDataset) -> np.float32:
    try:
        return np.float32(ds[0x5200, 0x9229][0][0x0028, 0x9132][0][0x0028, 0x1051].value)
    except KeyError:
        raise ValueError(
            "Window Width information not found in DICOM metadata.")


def get_data_file(set_type):
    return os.path.join(script_dir, f'../tmp/{set_type}/BCS-DBT-file-paths-{set_type}-v2.csv')


def get_files(set_type):
    return [
        os.path.join(
            script_dir, f'../tmp/{set_type}/BCS-DBT-file-paths-{set_type}-v2.csv'),
        os.path.join(
            script_dir, f'../tmp/{set_type}/BCS-DBT-labels-{set_type}-v2.csv'),
        os.path.join(
            script_dir, f'../tmp/{set_type}/BCS-DBT-boxes-{set_type}-v2.csv')
    ]
