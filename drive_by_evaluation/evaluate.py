import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.optimizers import SGD
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold

from drive_by_evaluation.db_machine_learning.multi_scorer import MultiScorer
from drive_by_evaluation.measure_collection import MeasureCollection
from drive_by_evaluation.db_machine_learning.db_data_set import DataSet
from drive_by_evaluation.deep_learning_keras.lstm import create_lstm_model
from drive_by_evaluation.deep_learning_keras.conv import create_conv_model

import operator

from drive_by_evaluation.db_machine_learning.confusion_matrix_util import print_confusion_matrix_measures, sumup_confusion_matrices
from drive_by_evaluation.parking_map_clustering.dbscan_clustering_directional import create_parking_space_map, filter_parking_space_map_mcs
from drive_by_evaluation.deep_learning_keras.evaluate_keras import simple_dense_model, predict_softmax
import time

from drive_by_evaluation.db_machine_learning.models import create_random_forest, predict


class DriveByEvaluation:

    # def __init__(self):
    #     self.clf_and_datasets = []

    def evaluate(self, create_and_train_model, predict_from_model, dataset, number_of_splits=5, shuffle=False):
        kfold = KFold(n_splits=number_of_splits, shuffle=shuffle)
        confusion_res = []
        for train, test in kfold.split(dataset.x, dataset.y_true):
            x_train = [x for i, x in enumerate(dataset.x) if i in train]
            y_train = [x for i, x in enumerate(dataset.y_true) if i in train]
            x_test = [x for i, x in enumerate(dataset.x) if i in test]
            y_test = [x for i, x in enumerate(dataset.y_true) if i in test]

            model = create_and_train_model(dataset, x_train, y_train)
            y_pred, y_true = predict_from_model(model, x_test, y_test)

            #print(y_true)
            #print(y_pred)
            labels = dataset.class_labels
            if dataset.is_softmax_y:
                labels = range(0, len(dataset.class_labels))
            confusion_m = confusion_matrix(y_true, y_pred, labels=labels)
            print_confusion_matrix_measures(confusion_m)
            confusion_res.append(confusion_m)

        confusion_sum = sumup_confusion_matrices(confusion_res, dataset.get_nr_of_classes())
        print_confusion_matrix_measures(confusion_sum)

        return confusion_sum


if __name__ == '__main__':
    base_path = 'C:\\sw\\master\\collected data\\'

    options = {
        'mc_min_speed': 3.0,
        'mc_merge': True,
        'mc_separation_threshold': 1.0,
        'mc_min_measure_count': 2,
        # 'mc_surrounding_times_s': [2.0, 5.0],
        'outlier_threshold_distance': 1.0,
        'outlier_threshold_diff': 0.5,
        # 'replacement_values': {0.01: 10.01},
        'min_measurement_value': 0.06,
    }

    dataset_softmax_10cm = None
    dataset_normal = None
    measure_collections_files_dir = MeasureCollection.read_directory(base_path, options=options)

    # parking_space_map_clusters, _ = create_parking_space_map(measure_collections_files_dir)
    # measure_collections_files_dir = filter_parking_space_map_mcs(measure_collections_files_dir,
    #                                                             parking_space_map_clusters)

    measure_collections_dir = {}
    for file_name, measure_collections in measure_collections_files_dir.items():
        print(file_name)
        dataset_softmax_10cm = DataSet.get_raw_sensor_dataset_per_10cm(measure_collections, dataset=dataset_softmax_10cm, is_softmax_y=True)
        dataset_normal = DataSet.get_dataset(measure_collections, dataset=dataset_normal)
        measure_collections_dir.update(MeasureCollection.mc_list_to_dict(measure_collections))

    start = time.time()
    # confusion_m_simp = evaluate_model(simple_dense_model, dataset)

    evaluator = DriveByEvaluation()
    # confusion_m_lstm = evaluator.evaluate(create_random_forest, predict, dataset_normal)
    confusion_m_lstm = evaluator.evaluate(simple_dense_model, predict_softmax, dataset_softmax_10cm)
    # confusion_m_conv = evaluate_model(create_conv_model, dataset)
    print(time.time() - start)

    # print_confusion_matrix_measures(confusion_m_simp)
    print('lstm')
    print_confusion_matrix_measures(confusion_m_lstm)
    # print('conv')
    # print_confusion_matrix_measures(confusion_m_conv)
