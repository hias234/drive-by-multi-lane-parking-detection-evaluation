import os
import random
import time

# from custom_clf import SurroundingClf

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
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
from drive_by_evaluation.db_machine_learning.models import create_random_forest, predict


def create_staged_forest(model, dataset, x_train, y_train, x_test, y_test, y_pred):
    derived_dataset = DataSet(dataset.class_labels, dataset.is_softmax_y)

