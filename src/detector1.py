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
  evidence = [row['evidence'] for row in data]
  labels = [row['labels'] for row in data]

  x_training, x_testing, y_training, y_testing  = train_test_split(evidence, labels, test_size=0.5)
  model.fit(x_training, y_training)
  predictions = model.predict(x_testing)

  correct = (y_testing == predictions).sum()
  incorrect = (y_testing != predictions).sum()
  total = len(predictions)

  print(f"Results for model: {type(model).__name__}")
  print(f"Correct: {correct}")
  print(f"Incorrect: {incorrect}")
  print(f"Accuracy: {100 * correct / total:.2f}%")
