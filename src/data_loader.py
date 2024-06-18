import os
import struct
import pandas as pd
import numpy as np
from skimage import io
from array import array

class DataLoader(object):
    def __init__(self):
        self.test_lbl_fname = 'tmp/test/BCS-DBT-labels-test-v2.csv'
        self.train_lbl_fname = 'tmp/train/BCS-DBT-labels-train-v2.csv'

    def load_training(self):
        data = pd.read_csv(self.train_lbl_fname)
        images = self.load_images('train', data['PatientID'], data['StudyUID'], data['View'])
        labels = self.convert_labels(data)
        return images, labels, data['PatientID']


    def load_testing(self):
        data = pd.read_csv(self.test_lbl_fname)
        images = self.load_images('test', data['PatientID'], data['StudyUID'], data['View'])
        labels = self.convert_labels(data)
        return images, labels, data['PatientID']

    def convert_labels(self, data):
        label_map = {'Normal': 0, 'Actionable': 1, 'Benign': 2, 'Cancer': 3}
        labels = []
        for _, row in data.iterrows():
            if row['Normal'] == 1:
                labels.append(label_map['Normal'])
            elif row['Actionable'] == 1:
                labels.append(label_map['Actionable'])
            elif row['Benign'] == 1:
                labels.append(label_map['Benign'])
            elif row['Cancer'] == 1:
                labels.append(label_map['Cancer'])
        return np.array(labels)

    def load_images(self, set_type, patient_ids, study_uids, views):
        images = []
        for pid, suid, view in zip(patient_ids, study_uids, views):
            image = self.load_image(set_type, pid, view)
            images.append(image)
        return images

    def load_image(self, set_type, patient_id, view):
        img_path = f"tmp/imgs/{set_type}/{patient_id}-{view}.png"
        img = io.imread(img_path)
        return img
