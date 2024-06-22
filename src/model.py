import pickle
import numpy as np
import pandas as pd
from sklearn import svm
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score, classification_report

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
    def __init__(self, type):
        setup_save_directory()
        plt.style.use("ggplot")
        self.model_type = type
        self.logger = setup_logger(type)
        self.classifier = self.get_model_type()
        self.train()
        self.add_train_summary()

    def add_train_summary(self):
        csv_file = f'{base_dir}/../tmp/model_summaries.csv'
        df = pd.read_csv(csv_file)
        new_data = {
            'Model': [self.model_type],
            'Accuracy': [f'{self.accuracy:.2f}'],
            'Images(#)': [self.train_dataset_length],
            'Precision': [f'{self.precision:.2f}'],
            'Recall': [f'{self.recall:.2f}'],
            'F1-score': [f'{self.f1:.2f}']
        }
        df_new = pd.DataFrame(new_data)
        df = pd.concat([df, df_new], ignore_index=True)
        df.to_csv(csv_file, index=False)
        print(f"Data appended and written back to {csv_file}.")

    def get_model_type(self):
        # These don't use epochs.
        # RandomForestClassifier, KNeighborsClassifier
        if self.model_type == "KNN":
            # KNN doesn't have confidence score.
            self.logger.info(
                "KNearestNeighbors with n_neighbors = 5, algorithm = auto, n_jobs = 10")
            return KNeighborsClassifier(n_neighbors=5, algorithm="auto", n_jobs=10)
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

    def render_confusion_matrix(self, confidence, y_pred, accuracy, conf_mat):
        self.logger.info(f"Trained Classifier Confidence: {confidence}")
        self.logger.info(f"Predicted Values: {y_pred}")
        self.logger.info(f"Accuracy of Classifier on Validation Image Data: {accuracy}")
        self.logger.info(f"Confusion Matrix for Validation Data: \n{conf_mat}")

        plt.matshow(conf_mat)
        plt.title("Confusion Matrix for Validation Data")
        plt.colorbar()
        plt.ylabel("True label")
        plt.xlabel("Predicted label")
        plt.savefig(plot_file_name(self.model_type, "validation"))
        plt.clf()

        self.logger.info(f"Making Predictions on Test Input Images: {self.test_labels_pred}")
        test_accuracy = accuracy_score(self.test_labels, self.test_labels_pred)
        self.logger.info(f"Calculating Accuracy of Trained Classifier on Test Data: {test_accuracy}")

        self.logger.info("Creating Confusion Matrix for Test Data...")
        conf_mat_test = confusion_matrix(self.test_labels, self.test_labels_pred)

        self.logger.info(f"Predicted Labels for Test Images: {self.test_labels_pred}")
        self.logger.info(f"Accuracy of Classifier on Test Images: {test_accuracy}")
        self.logger.info(f"Confusion Matrix for Test Data: \n{conf_mat_test}")

        plt.matshow(conf_mat_test)
        plt.title("Confusion Matrix for Test Data")
        plt.colorbar()
        plt.ylabel("True label")
        plt.xlabel("Predicted label")
        plt.savefig(plot_file_name(self.model_type, "test"))
        plt.clf()

        num_samples = min(len(self.test_img), 20)
        indices = np.random.randint(0, len(self.test_img), num_samples)

        for _, i in enumerate(indices):
            if i >= len(self.test_img):
                continue
            image_data = self.test_img[i]
            label = self.test_labels[i]
            predicted_label = self.test_labels_pred[i]
            patient_id = self.test_patient_ids[i]

            if len(image_data.shape) == 3 and image_data.shape[0] == 3:
                image_data = np.transpose(image_data, (1, 2, 0))

            plt.imshow(image_data, cmap='gray')
            plt.title(
                f"PatientID: {patient_id}\nLabeled Actual: {label}\nLabel Predicted: {predicted_label}", fontsize=8, color='blue')

            plt.colorbar()
            filename = image_file_name(
                self.model_type, patient_id, label, predicted_label)
            plt.savefig(filename)
            plt.clf()

    def train(self):
        self.logger.info("Training Starting...")

        self.logger.info("Loading Training Data Set...")
        train_dataset = DataLoader('tmp/train/BCS-DBT-labels-train-v2.csv', 'train')
        self.train_dataset_length = len(train_dataset)
        train_loader = TorchDataLoader(train_dataset, batch_size=2, shuffle=True, num_workers=4)

        self.logger.info("Loading Testing Data Set...")
        test_dataset = DataLoader('tmp/test/BCS-DBT-labels-test-v2.csv', 'test')
        test_loader = TorchDataLoader(test_dataset, batch_size=2, shuffle=False, num_workers=4)

        img_train, labels_train, _ = self.load_full_data(train_loader)
        img_test, labels_test, test_patient_ids = self.load_full_data(test_loader)

        self.test_img = img_test
        self.test_labels = labels_test
        self.test_patient_ids = test_patient_ids

        self.logger.info(f"Shape of training images: {img_train.shape}")
        self.logger.info(f"Shape of training labels: {labels_train.shape}")
        self.logger.info(f"Shape of test images: {self.test_img.shape}")
        self.logger.info(f"Shape of test labels: {self.test_labels.shape}")
        self.logger.info(f"Training label distribution: {np.bincount(labels_train)}")
        self.logger.info(f"Testing label distribution: {np.bincount(self.test_labels)}")

        x = img_train
        y = labels_train

        self.logger.info(f"Original shape of x: {x.shape}")
        self.logger.info(f"Original shape of y: {y.shape}")

        x_flat = x.reshape(x.shape[0], -1)
        self.logger.info(f"Reshaped x: {x_flat.shape}")
        y_flat = y

        x_train, x_test, y_train, y_test = train_test_split(x_flat, y_flat, test_size=0.1, stratify=y_flat)
        self.classifier.fit(x_train, y_train)
        self.create_pickle()

        self.accuracy = self.classifier.score(x_test, y_test)
        self.logger.info(f"Model accuracy: {self.accuracy}")

        y_pred = self.classifier.predict(x_test)
        self.precision = precision_score(y_test, y_pred, average='macro')
        self.recall = recall_score(y_test, y_pred, average='macro')
        self.f1 = f1_score(y_test, y_pred, average='macro')

        self.logger.info(f"Precision: {self.precision:.2f}")
        self.logger.info(f"Recall: {self.recall:.2f}")
        self.logger.info(f"F1-score: {self.f1:.2f}")

        test_img_flat = self.test_img.reshape(self.test_img.shape[0], -1)
        self.test_labels_pred = self.classifier.predict(test_img_flat)
        self.logger.info(f"Predicted labels for test image: {self.test_labels_pred}")

        self.logger.info("Calculating Classification Report...")
        classification_rep = classification_report(self.test_labels, self.test_labels_pred)
        self.logger.info(f"Classification Report:\n{classification_rep}")

        self.logger.info("Calculating Accuracy of trained Classifier...")
        confidence = self.classifier.score(x_test, y_test)
        y_pred = self.classifier.predict(x_test)
        accuracy = accuracy_score(y_test, y_pred)
        conf_mat = confusion_matrix(y_test, y_pred)
        self.render_confusion_matrix(confidence, y_pred, accuracy, conf_mat)
        self.logger.info("Training done")

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
