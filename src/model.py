import pickle
import numpy as np
from sklearn import svm
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score

from torch.utils.data import DataLoader as TorchDataLoader

from .data_loader import DataLoader

from .utils import (
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
        self.type = type
        self.logger = setup_logger(f"{type}-summary.log")
        self.classifier = self.get_model_type()
        self.train()

    def get_model_type(self):
        # These don't use epochs.
        # RandomForestClassifier, KNeighborsClassifier
        if self.type == "KNN":
            # KNN doesn't have confidence score.
            self.logger.info("KNearestNeighbors with n_neighbors = 5, algorithm = auto, n_jobs = 10")
            return KNeighborsClassifier(n_neighbors=5, algorithm="auto", n_jobs=10)
        elif self.type == "SVM":
            # Has confidence score.
            self.logger.info("SupportVectorMachines with gamma=0.1, kernel='poly'")
            return svm.SVC(gamma=0.1, kernel="poly")
        else:
            self.logger.info("RandomForestClassifier with n_estimators=100, random_state=42")
            return RandomForestClassifier(n_estimators=100, random_state=42)

    def create_pickle(self):
        # ckpt - old, pickle allows embedding malicious code
        # safetensor -
        with open(f'tmp/models/{self.type}_DBT.pickle', 'wb') as f:
            pickle.dump(self.classifier, f)
        pickle_in = open(f'tmp/models/{self.type}_DBT.pickle', 'rb')
        self.classifier = pickle.load(pickle_in)

    def render_confusion_matrix(self, confidence, y_pred, accuracy, conf_mat):
        self.logger.info(f"Trained Classifier Confidence: {confidence}")
        self.logger.info(f"Predicted Values: {y_pred}")
        self.logger.info(f"Accuracy of Classifier on Validation Image Data: {accuracy}")
        self.logger.info(f"Confusion Matrix: \n{conf_mat}")

        plt.matshow(conf_mat)
        plt.title("Confusion Matrix for Validation Data")
        plt.colorbar()
        plt.ylabel("True label")
        plt.xlabel("Predicted label")
        plt.savefig(plot_file_name(self.type, "validation"))

        self.logger.info(f"Making Predictions on Test Input Images: {self.test_labels_pred}")
        self.logger.info(f"Calculating Accuracy of Trained Classifier on Test Data: {accuracy}")

        self.logger.info("Creating Confusion Matrix for Test Data...")
        conf_mat_test = confusion_matrix(self.test_labels, self.test_labels_pred)

        self.logger.info(f"Predicted Labels for Test Images: {self.test_labels_pred}")
        self.logger.info(f"Accuracy of Classifier on Test Images: {accuracy}")
        self.logger.info(f"Confusion Matrix for Test Data: \n{conf_mat_test}")
        plt.matshow(conf_mat_test)
        plt.title("Confusion Matrix for Test Data")
        plt.colorbar()
        plt.ylabel("True label")
        plt.xlabel("Predicted label")
        plt.savefig(plot_file_name(self.type, "test"))
        plt.clf()

        num_samples = min(len(self.test_img), 20)
        indices = np.random.randint(0, len(self.test_img), num_samples)

        for idx, i in enumerate(indices):
            if i >= len(self.test_img):
                continue
            image_data = self.test_img[i]
            label = self.test_labels[i]
            predicted_label = self.test_labels_pred[i]
            patient_id = self.test_patient_ids[i]

            plt.imshow(image_data)
            plt.title(f"PatientID: {patient_id}\nActual Label: {label}\nModel Predicted Label: {predicted_label}", fontsize=8, color='blue')

            plt.colorbar()
            filename = image_file_name(self.type, idx, label)
            plt.savefig(filename)
            plt.clf()

    def train(self):
        self.logger.info("Training Starting...")

        self.logger.info("Loading Training Data Set...")
        train_dataset = DataLoader('tmp/train/BCS-DBT-labels-train-v2.csv', 'train')
        train_loader = TorchDataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)

        self.logger.info("Loading Testing Data Set...")
        test_dataset = DataLoader('tmp/test/BCS-DBT-labels-test-v2.csv', 'test')
        test_loader = TorchDataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)

        # Convert loaded data to numpy arrays for logging
        img_train, labels_train, train_patient_ids = self.load_full_data(train_loader)
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

        accuracy = self.classifier.score(x_test, y_test)
        self.logger.info(f"Model accuracy: {accuracy}")

        y_pred = self.classifier.predict(x_test)
        precision = precision_score(y_test, y_pred, average='macro')
        recall = recall_score(y_test, y_pred, average='macro')
        f1 = f1_score(y_test, y_pred, average='macro')

        self.logger.info(f"Precision: {precision:.2f}")
        self.logger.info(f"Recall: {recall:.2f}")
        self.logger.info(f"F1-score: {f1:.2f}")

        test_img_flat = self.test_img.reshape(self.test_img.shape[0], -1)
        self.test_labels_pred = self.classifier.predict(test_img_flat)
        self.logger.info(f"Predicted labels for test image: {self.test_labels_pred}")

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
        batch_idx = 0
        for batch in data_loader:
            self.logger.info(f"Batch index: {batch_idx}")
            images.extend(batch['image'].numpy())
            labels.extend(batch['label'].numpy())
            patient_ids.extend(batch['patient_id'].numpy())
            batch_idx+=1

        return np.array(images), np.array(labels), np.array(patient_ids)

    def predict(self, data_to_predict):
        loaded_model = load_pickle('KNN')
        if loaded_model:
            predictions = loaded_model.predict(data_to_predict)
            self.logger.info(predictions)
        else:
            self.logger.info("Failed to load the model.")
