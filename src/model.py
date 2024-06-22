import time
import pickle
import numpy as np
import pandas as pd
from sklearn import svm
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, classification_report

from torch.utils.data import DataLoader as TorchDataLoader

from .data_loader import DataLoader

from .utils import (
    base_dir,
    load_pickle,
    plot_file_name,
    setup_logger,
    image_file_name,
    setup_save_directory,
)


class Model:
    def __init__(self, model_type):
        start_time = time.time()
        setup_save_directory()
        plt.style.use("ggplot")
        self.model_type = model_type
        self.logger = setup_logger(model_type)
        self.classifier = self.select_classifier()
        self.train()
        self.log_train_summary(start_time)

    def log_train_summary(self, start_time):
        end_time = time.time()
        duration = end_time - start_time
        minutes, seconds = divmod(duration, 60)
        self.logger.info(f"Training time: {int(minutes)}:{int(seconds):02d}")

        csv_file = f'{base_dir}/../tmp/model_summaries.csv'
        df = pd.read_csv(csv_file)
        new_data = {
            'Model': [self.model_type],
            'Accuracy': [f'{self.accuracy:.2f}'],
            'Trained Set(#)': [self.train_set_length],
            'Test Set(#)': [self.test_set_length],
            'Precision': [f'{self.precision:.2f}'],
            'Recall': [f'{self.recall:.2f}'],
            'F1-score': [f'{self.f1:.2f}'],
            'Time Elapsed': [f'{self.f1:.2f}']
        }
        df_new = pd.DataFrame(new_data)
        df = pd.concat([df, df_new], ignore_index=True)
        df.to_csv(csv_file, index=False)
        
        print(f"Data appended and written back to {csv_file}.")

    def select_classifier(self):
        # These don't use epochs.
        # RandomForestClassifier, KNeighborsClassifier
        if self.model_type == "KNN":
            # KNN doesn't have confidence score.
            self.logger.info(
                "KNearestNeighbors with n_neighbors = 5, algorithm = auto, n_jobs = 10")
            return KNeighborsClassifier(algorithm="auto", n_jobs=10)
        elif self.model_type == "SVM":
            # Has confidence score.
            self.logger.info(
                "SupportVectorMachines with gamma=0.1, kernel='poly'")
            return svm.SVC(gamma=0.1, kernel="poly")
        else:
            self.logger.info(
                "RandomForestClassifier with n_estimators=100, random_state=42")
            return RandomForestClassifier(n_estimators=100, random_state=42)

    def create_pickle(self):
        # ckpt - old, pickle allows embedding malicious code
        # safetensor
        with open(f'tmp/models/{self.model_type}_DBT.pickle', 'wb') as f:
            pickle.dump(self.classifier, f)
        pickle_in = open(f'tmp/models/{self.model_type}_DBT.pickle', 'rb')
        self.classifier = pickle.load(pickle_in)

    def render_matrix(self, set_type):
        matrix = confusion_matrix(self.test_labels, self.test_labels_pred)
        plt.matshow(matrix)
        plt.title(f"Confusion Matrix for {set_type.capitalize()} Data")
        plt.colorbar()
        plt.ylabel("True label")
        plt.xlabel("Predicted label")
        plt.savefig(plot_file_name(self.model_type, set_type))
        plt.clf()

    def render_sampled_test_imgs_with_labels(self):
        num_samples = min(len(self.test_imgs), 20)
        indices = np.random.randint(0, len(self.test_imgs), num_samples)
        for _, i in enumerate(indices):
            if i >= len(self.test_imgs):
                continue
            image_data = self.test_imgs[i]
            label = self.test_labels[i]
            predicted_label = self.test_labels_pred[i]
            patient_id = self.test_patient_ids[i]

            if len(image_data.shape) == 3 and image_data.shape[0] == 3:
                image_data = np.transpose(image_data, (1, 2, 0))

            plt.imshow(image_data, cmap='gray')
            title = f'PatientID: {patient_id}\nLabeled Actual: {label}\nLabel Predicted: {predicted_label}'
            plt.title(title, fontsize=8, color='blue')

            plt.colorbar()
            filename = image_file_name(
                self.model_type, patient_id, label, predicted_label)
            plt.savefig(filename)
            plt.clf()

    def load_dataset(self, set_type):
        self.logger.info(f"Loading {set_type.capitalize()} Data Set...")
        dataset = DataLoader(
            f'tmp/{set_type}/BCS-DBT-labels-{set_type}-v2.csv', set_type)
        length = len(dataset)
        loader = TorchDataLoader(
            dataset, batch_size=32, shuffle=True, num_workers=4)
        return [length, loader]

    def evaluate(self, x_test, y_test):
        self.accuracy = self.classifier.score(x_test, y_test)
        y_pred = self.classifier.predict(x_test)
        self.precision = precision_score(y_test, y_pred, average='macro')
        self.recall = recall_score(y_test, y_pred, average='macro')
        self.f1 = f1_score(y_test, y_pred, average='macro')

        self.logger.info(f"Model Accuracy: {self.accuracy:.2f}")
        self.logger.info(f"Precision: {self.precision:.2f}")
        self.logger.info(f"Recall: {self.recall:.2f}")
        self.logger.info(f"F1-score: {self.f1:.2f}")

        test_img_flat = self.test_imgs.reshape(self.test_imgs.shape[0], -1)
        self.test_labels_pred = self.classifier.predict(test_img_flat)
        self.logger.info(
            f"Test Set Predicted Labels: \n{self.test_labels_pred}")

        classification_rep = classification_report(
            self.test_labels, self.test_labels_pred)
        self.logger.info(f"Classification Report:\n{classification_rep}")

    def load_full_data(self, data_loader):
        images = []
        labels = []
        patient_ids = []
        for batch in data_loader:
            image_batch, label_batch, patient_id_batch = batch
            images.extend(image_batch.numpy())
            labels.extend(label_batch.numpy())
            patient_ids.extend(patient_id_batch)

        return np.array(images), np.array(labels), np.array(patient_ids)

    def predict(self, data_to_predict):
        loaded_model = load_pickle(self.model_type)
        if loaded_model:
            predictions = loaded_model.predict(data_to_predict)
            self.logger.info(predictions)
        else:
            self.logger.info("Failed to load the model.")

    def train(self):
        self.logger.info("Training Starting...")

        self.train_set_length, train_loader = self.load_dataset('train')
        self.test_set_length, test_loader = self.load_dataset('test')

        self.train_imgs, self.train_labels, _ = self.load_full_data(
            train_loader)
        self.test_imgs, self.test_labels, self.test_patient_ids = self.load_full_data(
            test_loader)

        self.logger.info(f"Shape of training images: {self.train_imgs.shape}")
        self.logger.info(
            f"Shape of training labels: {self.train_labels.shape}")

        self.logger.info(f"Shape of test images: {self.test_imgs.shape}")
        self.logger.info(f"Shape of test labels: {self.test_labels.shape}")

        self.logger.info(
            f"Training label distribution: {np.bincount(self.train_labels)}")
        self.logger.info(
            f"Testing label distribution: {np.bincount(self.test_labels)}")

        x_flat = self.train_imgs.reshape(self.train_imgs.shape[0], -1)
        y_flat = self.train_labels

        x_train, x_test, y_train, y_test = train_test_split(
            x_flat, y_flat, test_size=0.1, stratify=y_flat)
        self.classifier.fit(x_train, y_train)
        self.create_pickle()
        self.evaluate(x_test, y_test)
        self.render_matrix('test')
        self.render_sampled_test_imgs_with_labels()

        self.logger.info("Training done")
