import sys
import numpy as np
from matplotlib import style
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn import model_selection, svm, preprocessing
from sklearn.metrics import accuracy_score, confusion_matrix

import tensorflow as tf
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import SimpleRNN, Dense

from .seep_deep import SeepDeep

from .utils import (
    setup_save_directory,
    create_log_file,
    image_file_name,
    get_file_name,
    create_pickle,
)


def get_model_type(type):
    if type == "KNN":
        print("KNearestNeighbors with n_neighbors = 5, algorithm = auto, n_jobs = 10")
        return KNeighborsClassifier(n_neighbors=5, algorithm="auto", n_jobs=10)
    elif type == "SVM":
        print("SupportVectorMachines with gamma=0.1, kernel='poly'")
        return svm.SVC(gamma=0.1, kernel="poly")
    elif type == "RFC":
        print("RandomForestClassifier with n_estimators=100, random_state=42")
        return RandomForestClassifier(n_estimators=100, random_state=42)
    elif type == "MLP":
        print(
            "MLPClassifier with hidden_layer_sizes=(100,), max_iter=300, alpha=1e-4, solver='sgd', verbose=10, random_state=1, learning_rate_init=0.001"
        )
        return MLPClassifier(
            hidden_layer_sizes=(100,),
            max_iter=300,
            alpha=1e-4,
            solver="sgd",
            verbose=10,
            random_state=1,
            learning_rate_init=0.001,
        )
    elif type == "RNN":
        print(
            "RecurrentNeuralNetwork with 128, input_shape=(28, 28), activation='relu'), Dense(10, activation='softmax'"
        )
        clf = Sequential(
            [
                SimpleRNN(128, input_shape=(28, 28), activation="relu"),
                Dense(10, activation="softmax"),
            ]
        )
        clf.compile(
            optimizer="adam",
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )
        return clf
    elif type == "CNN":
        print(
            "ConvolutionalNeuralNetwork with n_neighbors=5, algorithm='auto', n_jobs=10"
        )
        return

def base_model(type, clf):
    setup_save_directory()
    style.use("ggplot")
    log_file = create_log_file(f"{type}-summary.log")
    sys.stdout = log_file

    print("Loading MNIST Data...")

    print("Loading Training Data...")
    data = SeepDeep()
    img_train, labels_train = data.load_training()
    train_img = np.array(img_train)
    train_labels = np.array(labels_train)

    print("Loading Testing Data...")
    img_test, labels_test = data.load_testing()
    test_img = np.array(img_test)
    test_labels = np.array(labels_test)

    x = train_img
    y = train_labels

    print("Preparing Classifier Training and Validation Data...")
    x_train, x_test, y_train, y_test = model_selection.train_test_split(
        x, y, test_size=0.1
    )

    if type == "RNN":
        x_train = x_train.reshape((-1, 28, 28))
        x_test = x_test.reshape((-1, 28, 28))

    clf.fit(x_train, y_train)
    clf = create_pickle(clf, type)

    print("Calculating Accuracy of trained Classifier...")
    y_pred, confidence = None, None
    if type == "RNN":
        _, accuracy = clf.evaluate(x_test, y_test)
        y_test_pred_probs = clf.predict(x_test)
        y_test_pred_classes = np.argmax(y_test_pred_probs, axis=1)
        conf_mat = confusion_matrix(y_test, y_test_pred_classes)
        test_img = test_img.reshape((-1, 28, 28))
        test_labels_pred_probs = clf.predict(test_img)
        test_labels_pred = np.argmax(test_labels_pred_probs, axis=1)
    else:
        confidence = clf.score(x_test, y_test)
        y_pred = clf.predict(x_test)
        accuracy = accuracy_score(y_test, y_pred)
        conf_mat = confusion_matrix(y_test, y_pred)
        test_labels_pred = clf.predict(test_img)

    print("Trained Classifier Confidence: ", confidence)
    print("Predicted Values: ", y_pred)
    print("Accuracy of Classifier on Validation Image Data: ", accuracy)
    print("Confusion Matrix: ", conf_mat)

    plt.matshow(conf_mat)
    plt.title("Confusion Matrix for Validation Data")
    plt.colorbar()
    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.savefig(get_file_name("validation", type))

    print(f"Making Predictions on Test Input Images: {test_labels_pred}")
    print(f"Calculating Accuracy of Trained Classifier on Test Data: {accuracy}")

    print("Creating Confusion Matrix for Test Data...")
    conf_mat_test = confusion_matrix(test_labels, test_labels_pred)

    print("Predicted Labels for Test Images: ", test_labels_pred)
    print("Accuracy of Classifier on Test Images: ", accuracy)
    print("Confusion Matrix for Test Data:", conf_mat_test)

    plt.matshow(conf_mat_test)
    plt.title("Confusion Matrix for Test Data")
    plt.colorbar()
    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.savefig(get_file_name("test", type))
    plt.clf()

    a = np.random.randint(1, 50, 20)
    for idx, i in enumerate(a):
        two_d = (np.reshape(test_img[i], (28, 28)) * 255).astype(np.uint8)
        plt.title(
            f"Original Label: {test_labels[i]}  Predicted Label: {test_labels_pred[i]}"
        )
        plt.imshow(two_d, interpolation="nearest", cmap="gray")
        filename = image_file_name(type, idx, test_labels[i])
        plt.savefig(filename)
        plt.clf()

    print("Done")
