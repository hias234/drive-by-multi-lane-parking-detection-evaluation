import os
import random
import time

# from custom_clf import SurroundingClf
from wsgiref.validate import check_input

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from mlxtend.classifier import StackingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import KFold

from drive_by_evaluation.db_machine_learning.db_data_set import DataSet
from drive_by_evaluation.db_machine_learning.multi_scorer import MultiScorer
from drive_by_evaluation.ground_truth import GroundTruthClass
from drive_by_evaluation.measure_collection import MeasureCollection
from drive_by_evaluation.measurement import Measurement

from drive_by_evaluation.parking_map_clustering.dbscan_clustering_directional import create_parking_space_map, filter_parking_space_map_mcs


def predict(model, x_test, y_test):
    predictions = model.predict(np.array(x_test))

    y_pred = []
    y_true = []

    for i, prediction in enumerate(predictions):
        y_pred.append(prediction)
        y_true.append(y_test[i])

    return y_pred, y_true


def random_forest_100_trees(dataset, x_train, y_train):
    clf = RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=42)
    clf.fit(x_train, y_train)

    return clf


def random_forest_100_trees_entropy(dataset, x_train, y_train):
    clf = RandomForestClassifier(criterion='entropy', n_estimators=100, n_jobs=-1, random_state=42)
    clf.fit(x_train, y_train)

    return clf


def random_forest_1000_trees(dataset, x_train, y_train):
    clf = RandomForestClassifier(n_estimators=1000, n_jobs=-1, random_state=42)
    clf.fit(x_train, y_train)

    return clf


def random_forest_1000_trees_entropy(dataset, x_train, y_train):
    clf = RandomForestClassifier(criterion='entropy', n_estimators=1000, n_jobs=-1, random_state=42)
    clf.fit(x_train, y_train)

    return clf


def decision_tree(dataset, x_train, y_train):
    clf = DecisionTreeClassifier()
    clf.fit(x_train, y_train)

    return clf


def decision_tree_entropy(dataset, x_train, y_train):
    clf = DecisionTreeClassifier(criterion='entropy')
    clf.fit(x_train, y_train)

    return clf


def decision_tree_minsamples_10(dataset, x_train, y_train):
    clf = DecisionTreeClassifier(min_samples_leaf=10)
    clf.fit(x_train, y_train)

    return clf


def decision_tree_entropy_minsamples_10(dataset, x_train, y_train):
    clf = DecisionTreeClassifier(criterion='entropy', min_samples_leaf=10)
    clf.fit(x_train, y_train)

    return clf


def mlp_100_hidden_layer_maxiter_1000000(dataset, x_train, y_train):
    clf = MLPClassifier(hidden_layer_sizes=100, max_iter=1000000, random_state=42)
    clf.fit(x_train, y_train)

    return clf


def mlp_100_hidden_layer_maxiter_1000000_early_stopping(dataset, x_train, y_train):
    clf = MLPClassifier(hidden_layer_sizes=100, max_iter=1000000, random_state=42, early_stopping=True)
    clf.fit(x_train, y_train)

    return clf


def mlp_100_hidden_layer_maxiter_10000000_minlearning_rate(dataset, x_train, y_train):
    clf = MLPClassifier(hidden_layer_sizes=100, max_iter=10000000, random_state=42, learning_rate_init=0.00001)
    clf.fit(x_train, y_train)

    return clf


def mlp_5x50_hidden_layer_maxiter_1000000(dataset, x_train, y_train):
    clf = MLPClassifier(hidden_layer_sizes=(50,50,50,50,50), max_iter=1000000, random_state=42)
    clf.fit(x_train, y_train)

    return clf


def naive_bayes(dataset, x_train, y_train):
    clf = GaussianNB()
    clf.fit(x_train, y_train)

    return clf


def kNN1(dataset, x_train, y_train):
    clf = KNeighborsClassifier(n_neighbors=1)
    clf.fit(x_train, y_train)

    return clf


def kNN3(dataset, x_train, y_train):
    clf = KNeighborsClassifier(n_neighbors=3)
    clf.fit(x_train, y_train)

    return clf


def kNN5(dataset, x_train, y_train):
    clf = KNeighborsClassifier(n_neighbors=5)
    clf.fit(x_train, y_train)

    return clf


def kNN9(dataset, x_train, y_train):
    clf = KNeighborsClassifier(n_neighbors=9)
    clf.fit(x_train, y_train)

    return clf


def kNN21(dataset, x_train, y_train):
    clf = KNeighborsClassifier(n_neighbors=21)
    clf.fit(x_train, y_train)

    return clf


def svm(dataset, x_train, y_train):
    clf = SVC()
    clf.fit(x_train, y_train)

    return clf

def svm_sigmoid(dataset, x_train, y_train):
    clf = SVC(kernel='sigmoid')
    clf.fit(x_train, y_train)

    return clf


def create_stacked(dataset, x_train, y_train):
    for i, y in enumerate(dataset.y_true):
        dataset.y_true[i] = dataset.class_labels.index(y)

    for i, y in enumerate(y_train):
        y_train[i] = dataset.class_labels.index(y)
    dataset.class_labels = range(0, len(dataset.class_labels))

    clf1 = RandomForestClassifier(n_estimators=1000, n_jobs=-1, random_state=42)
    clf2 = KNeighborsClassifier(n_neighbors=10)
    clf3 = GaussianNB()
    clf4 = MLPClassifier(activation='relu', max_iter=100000, hidden_layer_sizes=(50,50,50,50,50))
    clf5 = MLPClassifier(activation='relu', max_iter=1000000, hidden_layer_sizes=(500,500))
    clf6 = RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=42)
    clf_meta = LogisticRegression()
    clf = StackingClassifier(classifiers=[clf1, clf2, clf3, clf4, clf5, clf6],
                             meta_classifier=clf_meta,
                             use_probas=True)

    clf.fit(x_train, y_train)

    return clf
