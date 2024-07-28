import time
import pickle
import numpy as np
import pandas as pd
from sklearn import svm
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import DecisionBoundaryDisplay
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, ConfusionMatrixDisplay, confusion_matrix

from torch.utils.data import DataLoader as TorchLoader

from .data_loader import DataLoader

from .utils import (
    base_dir,
    save_plot,
    setup_logger,
    image_file_name,
    setup_save_directory,
)

class Model:
    def __init__(self, strategy, scope, dataset_name):
        setup_save_directory()
        plt.style.use("ggplot")
        self.dataset_name = dataset_name
        self.scope = scope
        self.strategy = strategy
        self.logger = setup_logger(strategy, scope)
        self.classifier = None
        self.train_imgs = None
        self.train_labels = None
        self.test_imgs = None
        self.test_labels = None
        self.class_names = ['Normal', 'Actionable', 'Benign', 'Cancer']
        self.label_map = {0: 'Normal',
                          1: 'Actionable', 2: 'Benign', 3: 'Cancer'}
        self.target_names = ['Normal 0', 'Actionable 1', 'Benign 2', 'Cancer 3']

    @classmethod
    def from_pickle(cls, filepath):
        with open(filepath, 'rb') as f:
            data = pickle.load(f)

        instance = cls(strategy=data['strategy'], scope=data['scope'])
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
            'Model': [self.strategy],
            'Accuracy': [f'{self.accuracy:.2f}'],
            'Trained Set(#)': [self.train_length],
            'Test Set(#)': [self.test_length],
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
        file = f'tmp/set_{set_type}/BCS-DBT-labels-{set_type}-v2-{self.scope}.csv'
        dataset = DataLoader(file, set_type)
        length = len(dataset)
        loader = TorchLoader(dataset, batch_size=16,
                             shuffle=True, num_workers=4)
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
        if self.strategy == "KNN":
            self.log(
                "KNearestNeighbors with n_neighbors = 5, algorithm = auto, n_jobs = 10")
            return KNeighborsClassifier(algorithm="auto", n_jobs=10, n_neighbors=5)
        elif self.strategy == "SVM":
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
            'scope': self.scope,
            'train_imgs': self.train_imgs,
            'train_labels': self.train_labels,
            'test_imgs': self.test_imgs,
            'test_labels': self.test_labels,
            'strategy': self.strategy
        }

        pickle_file = f'tmp/models/{self.strategy.lower()}_{self.scope.lower()}_classifier.pickle'

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
        cm = confusion_matrix(self.test_labels, self.predictions)
        self.log("\n\nConfusion Matrix:")
        self.log(f'\n{cm}')
        unique_classes = np.unique(np.concatenate(
            (self.test_labels, self.predictions)))

        if not set(self.label_map.keys()).issuperset(set(unique_classes)):
            print(f"Unique classes in data: {unique_classes}")
            print(f"Label mapping: {self.label_map}")
            raise ValueError(
                "The unique classes in data do not match the expected label mapping.")

        disp = ConfusionMatrixDisplay.from_predictions(
            self.test_labels, self.predictions, display_labels=self.class_names)
        disp.plot(cmap='viridis')
        plt.title(f"Confusion Matrix: {set_type} set")
        plt.savefig(save_plot(f'{self.strategy}-{self.scope}-confusion-matrix'))

    def render_sampled_test_imgs_with_labels(self):
        num_samples = min(len(self.test_imgs), 16)
        indices = np.random.randint(0, len(self.test_imgs), num_samples)
        for _, i in enumerate(indices):
            if i >= len(self.test_imgs):
                continue
            image_data = self.test_imgs[i]
            label = self.test_labels[i]
            predicted_label = self.predictions[i]
            patient_id = self.test_patient_ids[i]

            if len(image_data.shape) == 3 and image_data.shape[0] == 3:
                image_data = np.transpose(image_data, (1, 2, 0))

            plt.imshow(image_data, cmap='gray')
            title = f'PatientID: {patient_id}\nLabeled Actual: {label}\nLabel Predicted: {predicted_label}'
            plt.title(title, fontsize=8, color='blue')

            plt.colorbar()
            filename = image_file_name(
                self.strategy, patient_id, label, predicted_label)
            plt.savefig(filename)
            plt.clf()

    # Tried using ConfusionMatrixDisplay.from_estimator
    # def render_knn_plot(self):
    #     print("Shape of self.train_imgs:", self.train_imgs.shape)
    #     print("Shape of self.test_imgs:", self.test_imgs.shape)

    #     train_imgs_flat = self.train_imgs.reshape(len(self.train_imgs), -1)
    #     test_imgs_flat = self.test_imgs.reshape(len(self.test_imgs), -1)
    #     scaler = StandardScaler()
    #     train_imgs_flat = scaler.fit_transform(train_imgs_flat)
    #     test_imgs_flat = scaler.transform(test_imgs_flat)

    #     pca = PCA(n_components=2)
    #     x_train_pca = pca.fit_transform(train_imgs_flat)
    #     x_test_pca = pca.transform(test_imgs_flat)
    #     fig, axs = plt.subplots(ncols=2, figsize=(12, 5))

    #     for ax, weights in zip(axs, ("uniform", "distance")):
    #         self.classifier.set_params(weights=weights)
    #         self.classifier.fit(x_train_pca, self.train_labels)
    #         cm = ConfusionMatrixDisplay.from_estimator(
    #             self.classifier, x_test_pca, self.test_labels, ax=ax
    #         )
    #         ax.set_xlim(-1000, 200)
    #         ax.set_ylim(0, 500)

    #         ax.set_title(f"KNN decision boundaries\n(weights={weights!r})")

    #     plt.tight_layout()
    #     plt.savefig(save_plot(f'{self.strategy}-{self.scope}-xlim-ylim-graph'))


    def render_knn_plot(self):
        print("Shape of self.train_imgs:", self.train_imgs.shape)
        print("Shape of self.test_imgs:", self.test_imgs.shape)
        train_imgs_flat = self.train_imgs.reshape(len(self.train_imgs), -1)
        test_imgs_flat = self.test_imgs.reshape(len(self.test_imgs), -1)
        scaler = StandardScaler()
        train_imgs_flat = scaler.fit_transform(train_imgs_flat)
        test_imgs_flat = scaler.transform(test_imgs_flat)
        pca = PCA(n_components=2)
        x_train_pca = pca.fit_transform(train_imgs_flat)
        x_test_pca = pca.transform(test_imgs_flat)

        fig, axs = plt.subplots(ncols=2, figsize=(12, 5))

        for ax, weights in zip(axs, ("uniform", "distance")):
            self.classifier.set_params(weights=weights)
            self.classifier.fit(x_train_pca, self.train_labels)
            disp = DecisionBoundaryDisplay.from_estimator(
                self.classifier,
                x_train_pca,
                response_method="predict",
                plot_method="pcolormesh",
                shading="auto",
                ax=ax,
            )

            scatter = ax.scatter(x_test_pca[:, 0], x_test_pca[:, 1], c=self.test_labels, edgecolors="k")
            legend = ax.legend(
                scatter.legend_elements()[0],
                np.unique(self.test_labels),
                loc="lower left",
                title="Classes",
            )
            ax.set_xlim(-1000, 200)
            ax.set_ylim(-1000, 500)
            ax.set_title(f"KNN decision boundaries\n(weights={weights!r})")

        plt.tight_layout()
        plt.savefig(save_plot(f'{self.strategy}-{self.scope}-xlim-ylim-graph'))

    def evaluate(self):
        test_img_flat = self.test_imgs.reshape(self.test_imgs.shape[0], -1)
        self.predictions = self.classifier.predict(test_img_flat)
        self.accuracy = accuracy_score(self.test_labels, self.predictions)
        
        # ERROR: Full Dataset 
        # unique_classes = len(set(self.test_labels))
        # print(unique_classes)
        # if unique_classes == 4:
        #     target_names = ['Normal 0', 'Actionable 1', 'Benign 2', 'Cancer 3']
        # elif unique_classes == 3:
        #     target_names = ['Normal 0', 'Actionable 1', 'Benign 2']
        # else:
        #     raise ValueError(f"Unexpected number of classes: {unique_classes}")
        # ERROR: Full dataset
        # target_names = np.unique(np.concatenate(
        #     (self.test_labels, self.predictions)))
        
        report = classification_report(self.test_labels, self.predictions, target_names=self.target_names, output_dict=True)
        self.precision = report['macro avg']['precision']
        self.recall = report['macro avg']['recall']
        self.f1 = report['macro avg']['f1-score']
        
        self.log(f"Model Accuracy: {self.accuracy:.2f}")
        self.log(f"Precision: {self.precision:.2f}")
        self.log(f"Recall: {self.recall:.2f}")
        self.log(f"F1-score: {self.f1:.2f}")

        self.log(f"\n\nClassification Report:\n{classification_report(self.test_labels, self.predictions, target_names=['Normal 0', 'Actionable 1', 'Benign 2', 'Cancer 3'])}")

    def train(self):
        self.log("Training Starting...")
        self.train_length, train_loader = self.load_dataset(self.dataset_name)
        self.test_length, test_loader = self.load_dataset(self.dataset_name)

        self.train_imgs, self.train_labels, _ = self.load_full_data(
            train_loader)

        self.test_imgs, self.test_labels, self.test_patient_ids = self.load_full_data(
            test_loader)

        x_flat = self.train_imgs.reshape(self.train_imgs.shape[0], -1)
        y_flat = self.train_labels
        x_train, _, y_train, _ = train_test_split(
            x_flat, y_flat, stratify=y_flat)
        # x_train, x_test, y_train, y_test = train_test_split(
        #     x_flat, y_flat, test_size=0.1, stratify=y_flat)
        self.classifier.fit(x_train, y_train)
        self.evaluate()

        self.create_pickle()
