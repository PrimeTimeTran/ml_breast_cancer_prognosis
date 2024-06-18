import sys
import pickle
import numpy as np
from matplotlib import style
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn import model_selection, svm, preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score


from .data_loader import DataLoader

from .utils import (
    load_pickle,
    plot_file_name,
    create_log_file,
    image_file_name,
    setup_save_directory,
)

class Model:
    def __init__(self, type):
        style.use("ggplot")
        setup_save_directory()
        self.type = type
        sys.stdout = create_log_file(f"{type}-summary.log")
        self.classifier = self.get_model_type()
        self.train()

    def get_model_type(self):
        # These don't use epochs.
        # RandomForestClassifier, KNeighborsClassifier
        if self.type == "KNN":
            # KNN doesn't have confidence score.
            print("KNearestNeighbors with n_neighbors = 5, algorithm = auto, n_jobs = 10")
            return KNeighborsClassifier(n_neighbors=5, algorithm="auto", n_jobs=10)
        elif self.type == "SVM":
            # Has confidence score.
            print("SupportVectorMachines with gamma=0.1, kernel='poly'")
            return svm.SVC(gamma=0.1, kernel="poly")
        else:
            print("RandomForestClassifier with n_estimators=100, random_state=42")
            return RandomForestClassifier(n_estimators=100, random_state=42)

    def create_pickle(self):
        # ckpt - old, pickle allows embedding malicious code
        # safetensor -
        with open(f'tmp/models/{self.type}_DBT.pickle', 'wb') as f:
            pickle.dump(self.classifier, f)
        pickle_in = open(f'tmp/models/{self.type}_DBT.pickle', 'rb')
        self.classifier = pickle.load(pickle_in)

    def render_confusion_matrix(self, confidence, y_pred, accuracy, conf_mat):
        print("Trained Classifier Confidence: ", confidence)
        print("Predicted Values: ", y_pred)
        print("Accuracy of Classifier on Validation Image Data: ", accuracy)
        print("Confusion Matrix: ", conf_mat)

        plt.matshow(conf_mat)
        plt.title("Confusion Matrix for Validation Data")
        plt.colorbar()
        plt.ylabel("True label")
        plt.xlabel("Predicted label")
        plt.savefig(plot_file_name("validation", self.type))

        print(f"Making Predictions on Test Input Images: {self.test_labels_pred}")
        print(f"Calculating Accuracy of Trained Classifier on Test Data: {accuracy}")

        print("Creating Confusion Matrix for Test Data...")
        conf_mat_test = confusion_matrix(self.test_labels, self.test_labels_pred)

        print("Predicted Labels for Test Images: ", self.test_labels_pred)
        print("Accuracy of Classifier on Test Images: ", accuracy)
        print("Confusion Matrix for Test Data:", conf_mat_test)
        plt.matshow(conf_mat_test)
        plt.title("Confusion Matrix for Test Data")
        plt.colorbar()
        plt.ylabel("True label")
        plt.xlabel("Predicted label")
        plt.savefig(plot_file_name("test", self.type))
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
            # plt.show()
            plt.clf()

    def train(self):
        print("Training Starting...")
        print("Loading Training Data Set...")
        data = DataLoader()
        img_train, labels_train, _ = data.load_training()
        train_img = np.array(img_train)
        train_labels = np.array(labels_train)

        print(f"Shape of training images: {train_img.shape}")
        print(f"Shape of training labels: {train_labels.shape}")

        print("Loading Testing Data Set...")
        img_test, labels_test, test_patient_ids = data.load_testing()
        self.test_img = np.array(img_test)
        self.test_labels = np.array(labels_test)
        self.test_patient_ids = test_patient_ids

        print(f"Shape of test images: {self.test_img.shape}")
        print(f"Shape of test labels: {self.test_labels.shape}")

        print(f"Training label distribution: {np.bincount(train_labels)}")
        print(f"Testing label distribution: {np.bincount(self.test_labels)}")

        x = train_img
        y = train_labels

        print(f"Original shape of x: {x.shape}")
        print(f"Original shape of y: {y.shape}")

        x_flat = x.reshape(x.shape[0], -1)
        print(f"Reshaped x: {x_flat.shape}")
        y_flat = y

        x_train, x_test, y_train, y_test = train_test_split(x_flat, y_flat, test_size=0.1, stratify=y_flat)
        self.classifier.fit(x_train, y_train)
        self.create_pickle()

        accuracy = self.classifier.score(x_test, y_test)
        print(f"Model accuracy: {accuracy}")

        y_pred = self.classifier.predict(x_test)
        precision = precision_score(y_test, y_pred, average='macro')
        recall = recall_score(y_test, y_pred, average='macro')
        f1 = f1_score(y_test, y_pred, average='macro')

        print(f"Precision: {precision:.2f}")
        print(f"Recall: {recall:.2f}")
        print(f"F1-score: {f1:.2f}")

        test_img_flat = self.test_img.reshape(self.test_img.shape[0], -1)
        self.test_labels_pred = self.classifier.predict(test_img_flat)
        print(f"Predicted labels for test image: {self.test_labels_pred}")

        print("Calculating Accuracy of trained Classifier...")
        confidence = self.classifier.score(x_test, y_test)
        y_pred = self.classifier.predict(x_test)
        accuracy = accuracy_score(y_test, y_pred)
        conf_mat = confusion_matrix(y_test, y_pred)
        self.render_confusion_matrix(confidence, y_pred, accuracy, conf_mat)
        print("Training done")

    def predict(self, data_to_predict):
        loaded_model = load_pickle('KNN')
        if loaded_model:
            predictions = loaded_model.predict(data_to_predict)
            print(predictions)
        else:
            print("Failed to load the model.")
