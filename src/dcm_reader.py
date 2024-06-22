import os
import cv2
import pydicom
import numpy as np
import pandas as pd
from io import StringIO, BytesIO
from skimage.exposure import rescale_intensity
from typing import AnyStr, BinaryIO, Optional, Union

from .utils import base_dir

FilePathOrBuffer = Union[str, StringIO, BytesIO]

def view_image(set_type, df, plt, num):
    view_series = df.iloc[num]
    view = view_series["View"]
    image_path = os.path.join(base_dir, f"../tmp/{set_type}/manifest-1617905855234/", view_series["descriptive_path"])
    image = dcmread_image(fp=image_path, view=view)
    patient_id = view_series['PatientID']
    filename = f'{patient_id}-{view}.png'
    output_path = os.path.join(base_dir, f'../tmp/imgs/{set_type}/{filename}')
    print(f'PatientId: {patient_id}')
    print(f'View: {view}')
    resized_image = cv2.resize(image[0], (1996, 2457))
    write_to_disk(plt, output_path, resized_image)
    plt.imshow(resized_image, cmap=plt.cm.gray)
    return image

def write_to_disk(plt, output_path, resized_image):
    plt.imsave(output_path, resized_image)

def dcmread_image(
    fp: Union[str, "os.PathLike[AnyStr]", BinaryIO],
    view: str,
    index: Optional[int] = None,
) -> np.ndarray:
    """Read pixel array from DBT DICOM file"""
    print(f'Loading file: {fp}')
    ds = pydicom.dcmread(fp)
    ds.decompress(handler_name="pylibjpeg")
    pixel_array = ds.pixel_array

    if index is not None:
        pixel_array = pixel_array[index]

    view_laterality = view[0].upper()
    image_laterality = get_image_laterality(pixel_array)

    if image_laterality != view_laterality:
        pixel_array = np.flip(pixel_array, axis=(-1, -2))

    window_center = get_window_center(ds)
    window_width = get_window_width(ds)
    low = (2 * window_center - window_width) / 2
    high = (2 * window_center + window_width) / 2
    pixel_array = rescale_intensity(
        pixel_array, in_range=(low, high), out_range="dtype"
    )
    return pixel_array

def read_boxes(
    boxes_fp: FilePathOrBuffer, filepaths_fp: FilePathOrBuffer
) -> pd.DataFrame:
    """Read pandas DataFrame with bounding boxes joined with file paths"""
    df_boxes = pd.read_csv(boxes_fp)
    df_filepaths = pd.read_csv(filepaths_fp)
    primary_key = ("PatientID", "StudyUID", "View")
    if not all([key in df_boxes.columns for key in primary_key]):
        raise AssertionError(
            f"Not all primary key columns {primary_key} are present in bounding boxes columns {df_boxes.columns}"
        )
    if not all([key in df_boxes.columns for key in primary_key]):
        raise AssertionError(
            f"Not all primary key columns {primary_key} are present in file paths columns {df_filepaths.columns}"
        )
    return pd.merge(df_boxes, df_filepaths, on=primary_key)

def draw_box(
    image: np.ndarray,
    x: int,
    y: int,
    width: int,
    height: int,
    color: Optional[Union[int, tuple]] = None,
    lw=4,
):
    """Draw bounding box on the image"""
    x = min(max(x, 0), image.shape[1] - 1)
    y = min(max(y, 0), image.shape[0] - 1)
    if color is None:
        color = np.max(image)
    if len(image.shape) > 2 and not hasattr(color, "__len__"):
        color = (color,) + (0,) * (image.shape[-1] - 1)
    image[y : y + lw, x : x + width] = color
    image[y + height - lw : y + height, x : x + width] = color
    image[y : y + height, x : x + lw] = color
    image[y : y + height, x + width - lw : x + width] = color
    return image



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
