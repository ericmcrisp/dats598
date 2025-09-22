from utils.perceptron import Perceptron

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.utils import shuffle


def test_perceptron():
    df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data', header=None)
    df = shuffle(df)
    X = df.iloc[:, 0:4].values
    labels = df.iloc[:, 4].values
    unique_labels = set(labels)
    train_data, test_data, train_labels, test_labels = train_test_split(
                            X, labels, test_size=0.2)
    overall_accuracy = 0
    # since binary classification, test the flower individually
    for flower in unique_labels:
        flower_train_labels = np.where(train_labels == flower, 1, -1)
        flower_test_labels = np.where(test_labels == flower, 1, -1)
        p = Perceptron(input_dim=X.shape[1])
        p.fit(train_data, flower_train_labels)
        predictions = p.predict(test_data)
        results = accuracy_score(predictions, flower_test_labels)
        overall_accuracy += results
        print(f'{flower} Accuracy: \n', round(results, 3) * 100, "%")
    print('Average Accuracy: \n', round(overall_accuracy / len(unique_labels), 3) * 100, "%")
