import csv
import random
import sys
import pickle
import numpy as np
from matplotlib import style
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn import model_selection, svm, preprocessing
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score

from sklearn.svm import SVC
from sklearn.linear_model import Perceptron
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier

model = Perceptron()

with open("../tmp/train/BCS-DBT-file-paths-train-v2.csv") as f:
  reader = csv.reader(f)
  next(reader)
  data = []
  for row in reader:
    data.append({
      "evidence": [float(cell) for cell in row[:4]],
      "label": "Authenticate" if row[4] == "0" else "Counterfeit"
    })

  holdout = int(0.5 * len(data))
  random.shuffle(data)
  testing = data[:holdout]
  training = data[:holdout]

  x_training = [row['evidence'] for row in training]
  y_training = [row['label'] for row in training]
  model.fit(x_training, y_training)

  x_testing = [row['evidence'] for row in testing]
  y_testing = [row['label'] for row in testing]
  predictions = model.predict(x_testing)

  correct, incorrect, total = 0, 0, 0
  for actual, predicted in zip(y_testing, predictions):
    total += 1
    if actual == predicted:
      correct += 1
    else: 
      incorrect += 1

  print(f"Results for model: {type(model).__name__}")
  print(f"Correct: {correct}")
  print(f"Incorrect: {incorrect}")
  print(f"Accuracy: {100 * correct / total:.2f}%")
