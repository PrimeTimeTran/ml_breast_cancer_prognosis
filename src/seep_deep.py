import os
import struct
import pandas as pd
import numpy as np
from skimage import io
from array import array

from .utils import data_dir

class SeepDeep(object):
    def __init__(self):
        self.path = 'tmp/dataset'

        self.test_img_fname = 'tmp/imgs/test'
        self.test_lbl_fname = 'tmp/imgs/test'

        self.train_img_fname = 'tmp/imgs/train'
        self.train_lbl_fname = 'tmp/train/BCS-DBT-labels-train-v2.csv'

        self.test_images = []
        self.test_labels = []

        self.train_images = []
        self.train_labels = []

    def load_testing(self):
        ims, labels = self.load(os.path.join(self.path, self.test_img_fname),
                                os.path.join(self.path, self.test_lbl_fname))

        self.test_images = ims
        self.test_labels = labels

        return ims, labels

    def load_training(self):
        df = pd.read_csv(self.train_lbl_fname)

        img_train = []
        labels_train = []
        for _, row in df.iterrows():
            patient_id = row['PatientID']
            study_uid = row['StudyUID']
            view = row['View']

            img_path = f"{self.train_img_fname}/{patient_id}-{view}.png"

            img = io.imread(img_path)
            img_train.append(img)
            labels = [row['Normal'], row['Actionable'], row['Benign'], row['Cancer']]
            labels_train.append(labels)

        img_train = np.array(img_train)
        labels_train = np.array(labels_train)

        return img_train, labels_train

    @classmethod
    def load(cls, path_img, path_lbl):
        with open(path_lbl, 'rb') as file:
            magic, size = struct.unpack(">II", file.read(8))
            if magic != 2049:
                raise ValueError('Magic number mismatch, expected 2049,'
                                 'got {}'.format(magic))

            labels = array("B", file.read())

        with open(path_img, 'rb') as file:
            magic, size, rows, cols = struct.unpack(">IIII", file.read(16))
            if magic != 2051:
                raise ValueError('Magic number mismatch, expected 2051,'
                                 'got {}'.format(magic))

            image_data = array("B", file.read())

        images = []
        for i in range(size):
            images.append([0] * rows * cols)

        for i in range(size):
            images[i][:] = image_data[i * rows * cols:(i + 1) * rows * cols]

        return images, labels

    @classmethod
    def display(cls, img, width=28, threshold=200):
        render = ''
        for i in range(len(img)):
            if i % width == 0:
                render += '\n'
            if img[i] > threshold:
                render += '@'
            else:
                render += '.'
        return render
