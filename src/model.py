import time
import pickle
import numpy as np
import pandas as pd
from sklearn import svm
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import DecisionBoundaryDisplay
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, classification_report

from torch.utils.data import DataLoader as TorchDataLoader

from .data_loader import DataLoader

from .utils import (
    base_dir,
    plot_file_name,
    setup_logger,
    image_file_name,
    setup_save_directory,
    plot_graph_name,
)


class Model:
    def __init__(self, model_type, train_scope):
        setup_save_directory()
        plt.style.use("ggplot")
        self.train_scope = train_scope
        self.model_type = model_type
        self.logger = setup_logger(model_type)
        self.classifier = None
        self.train_imgs = None
        self.train_labels = None
        self.test_imgs = None
        self.test_labels = None

    @classmethod
    def from_pickle(cls, filepath):
        with open(filepath, 'rb') as f:
            data = pickle.load(f)

        instance = cls(model_type=data['model_type'], train_scope=data['train_scope'])
        instance.classifier = data.get('classifier')
        instance.train_imgs = data.get('train_imgs')
        instance.train_labels = data.get('train_labels')
        instance.test_imgs = data.get('test_imgs')
        instance.test_labels = data.get('test_labels')

        return instance

    def log(self, value):
        self.logger.info(value)

    def log_train_summary(self, start_time):
        end_time = time.time()
        duration = end_time - start_time
        minutes, seconds = divmod(duration, 60)
        time_elapsed = f'{int(minutes)}:{int(seconds):02d}'
        self.log(f"Training time: {time_elapsed}")

        csv_file = f'{base_dir}/../tmp/training_summaries.csv'
        df = pd.read_csv(csv_file)
        new_data = {
            'Model': [self.model_type],
            'Accuracy': [f'{self.accuracy:.2f}'],
            'Trained Set(#)': [self.train_set_length],
            'Test Set(#)': [self.test_set_length],
            'Precision': [f'{self.precision:.2f}'],
            'Recall': [f'{self.recall:.2f}'],
            'F1-score': [f'{self.f1:.2f}'],
            'Time Elapsed': [f'{time_elapsed}']
        }
        df_new = pd.DataFrame(new_data)
        df = pd.concat([df, df_new], ignore_index=True)
        df.to_csv(csv_file, index=False)

        print(f"Data appended and written back to {csv_file}.")

    def load_dataset(self, set_type):
        self.log(f"Loading {set_type.capitalize()} Data Set...")
        dataset = DataLoader(
            f'tmp/{set_type}/BCS-DBT-labels-{set_type}-v2-{self.train_scope}.csv', set_type)
        length = len(dataset)
        loader = TorchDataLoader(
            dataset, batch_size=32, shuffle=True, num_workers=4)
        return [length, loader]

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

    def select_classifier(self):
        if self.model_type == "KNN":
            self.log(
                "KNearestNeighbors with n_neighbors = 5, algorithm = auto, n_jobs = 10")
            return KNeighborsClassifier(algorithm="auto", n_jobs=10)
        elif self.model_type == "SVM":
            self.log(
                "SupportVectorMachines with gamma=0.1, kernel='poly'")
            return svm.SVC(gamma=0.1, kernel="poly")
        else:
            self.log(
                "RandomForestClassifier with n_estimators=100, random_state=42")
            return RandomForestClassifier(n_estimators=100, random_state=42)

    def create_pickle(self):
        self.log("Creating Pickle...")
        data = {
            'train_scope': self.train_scope,
            'train_imgs': self.train_imgs,
            'train_labels': self.train_labels,
            'test_imgs': self.test_imgs,
            'test_labels': self.test_labels,
            'model_type': self.model_type
        }

        pickle_file = f'tmp/models/{self.model_type.lower()}_{self.train_scope.lower()}_classifier.pickle'

        with open(pickle_file, 'wb') as f:
            pickle.dump(data, f)

        with open(pickle_file, 'rb') as f:
            loaded_data = pickle.load(f)

            assert (loaded_data['train_imgs'] ==
                    self.train_imgs).all(), "Train images mismatch"
            assert (loaded_data['train_labels'] ==
                    self.train_labels).all(), "Train labels mismatch"
            assert (loaded_data['test_imgs'] ==
                    self.test_imgs).all(), "Test images mismatch"
            assert (loaded_data['test_labels'] ==
                    self.test_labels).all(), "Test labels mismatch"

        self.log(
            f"Successfully created and verified pickle file: {pickle_file}")

    def fit_classifier(self):
        self.classifier.fit(self.train_imgs.reshape(
            self.train_imgs.shape[0], -1), self.train_labels)

    def render_matrix(self, set_type):
        matrix = confusion_matrix(self.test_labels, self.test_labels_pred)
        plt.matshow(matrix)
        plt.title(f"Confusion Matrix for {set_type.capitalize()} Data")
        plt.colorbar()
        plt.ylabel("True label")
        plt.xlabel("Predicted label")
        plt.savefig(plot_file_name(self.model_type, self.train_scope, set_type))
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

    def render_knn_plot(self):
        self.logger.info("Plotting KNN...")
        train_imgs_flat = self.train_imgs.reshape(len(self.train_imgs), -1)
        test_imgs_flat = self.test_imgs.reshape(len(self.test_imgs), -1)
        pca = PCA(n_components=2)
        x_train_pca = pca.fit_transform(train_imgs_flat)
        x_test_pca = pca.transform(test_imgs_flat)
        
        _, axs = plt.subplots(ncols=2, figsize=(12, 5))
        
        for ax, weights in zip(axs, ("uniform", "distance")):
            self.classifier.set_params(weights=weights)
            self.classifier.fit(x_train_pca, self.train_labels)
            disp = DecisionBoundaryDisplay.from_estimator(
                self.classifier,
                x_test_pca,
                response_method="predict",
                plot_method="pcolormesh",
                shading="auto",
                alpha=0.5,
                ax=ax,
            )
            scatter = disp.ax_.scatter(x_test_pca[:, 0], x_test_pca[:, 1], c=self.test_labels, edgecolors="k")
            disp.ax_.legend(
                scatter.legend_elements()[0],
                np.unique(self.test_labels),
                loc="lower left",
                title="Classes",
            )
            ax.set_title(f"KNN decision boundaries\n(weights={weights!r})")
        plt.savefig(plot_graph_name(self.model_type, self.train_scope))

    def evaluate(self, x_test, y_test):
        self.accuracy = self.classifier.score(x_test, y_test)
        y_pred = self.classifier.predict(x_test)
        self.precision = precision_score(y_test, y_pred, average='macro')
        self.recall = recall_score(y_test, y_pred, average='macro')
        self.f1 = f1_score(y_test, y_pred, average='macro')

        self.log(f"Model Accuracy: {self.accuracy:.2f}")
        self.log(f"Precision: {self.precision:.2f}")
        self.log(f"Recall: {self.recall:.2f}")
        self.log(f"F1-score: {self.f1:.2f}")

        test_img_flat = self.test_imgs.reshape(self.test_imgs.shape[0], -1)
        self.test_labels_pred = self.classifier.predict(test_img_flat)
        self.log(
            f"Test Set Predicted Labels: \n{self.test_labels_pred}")

        classification_rep = classification_report(
            self.test_labels, self.test_labels_pred)
        self.log(f"Classification Report:\n{classification_rep}")

    def train(self):
        self.log("Training Starting...")

        self.train_set_length, train_loader = self.load_dataset('train')
        self.test_set_length, test_loader = self.load_dataset('test')

        self.train_imgs, self.train_labels, _ = self.load_full_data(
            train_loader)

        self.test_imgs, self.test_labels, self.test_patient_ids = self.load_full_data(
            test_loader)

        self.log(f"Shape of training images: {self.train_imgs.shape}")
        self.log(f"Shape of training labels: {self.train_labels.shape}")

        self.log(f"Shape of test images: {self.test_imgs.shape}")
        self.log(f"Shape of test labels: {self.test_labels.shape}")

        self.log(
            f"Training label distribution: {np.bincount(self.train_labels)}")
        self.log(
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

        self.render_knn_plot()
