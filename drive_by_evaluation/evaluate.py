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

from drive_by_evaluation.db_machine_learning.models import create_random_forest, predict, create_stacked, create_decision_tree, create_mlp


class DriveByEvaluation:

    # def __init__(self):
    #     self.clf_and_datasets = []

    def evaluate(self, create_and_train_model, predict_from_model, dataset, post_process=None, number_of_splits=5, shuffle=False):
        kfold = KFold(n_splits=number_of_splits, shuffle=shuffle, random_state=42)

        predictions = ['' for i in range(len(dataset.x))]
        print(len(predictions))

        fold_nr = 1
        confusion_res = []
        for train, test in kfold.split(dataset.x, dataset.y_true):
            print('fold', fold_nr)
            x_train = [x for i, x in enumerate(dataset.x) if i in train]
            y_train = [x for i, x in enumerate(dataset.y_true) if i in train]

            #model, dataset = create_and_train_model(dataset, x_train, y_train)
            model = create_and_train_model(dataset, x_train, y_train)
            print('trained')

            x_test = [x for i, x in enumerate(dataset.x) if i in test]
            y_test = [x for i, x in enumerate(dataset.y_true) if i in test]

            y_pred, y_true = predict_from_model(model, x_test, y_test)
            print('predicted')

            if post_process is not None:
                y_pred = post_process(model, dataset, x_train, y_train, x_test, y_test, y_pred)

            for i, test_index in enumerate(test):
                    predictions[test_index] = y_pred[i]

            #print(y_true)
            #print(y_pred)
            labels = dataset.class_labels
            if dataset.is_softmax_y:
                labels = range(0, len(dataset.class_labels))
            confusion_m = confusion_matrix(y_true, y_pred, labels=labels)
            print_confusion_matrix_measures(confusion_m)
            confusion_res.append(confusion_m)
            fold_nr += 1

        confusion_sum = sumup_confusion_matrices(confusion_res, dataset.get_nr_of_classes())
        print_confusion_matrix_measures(confusion_sum)

        return confusion_sum, predictions


def enhance_dataset2(dataset, predictions):
    dataset_normal_plus = DataSet(class_labels=dataset.class_labels, is_softmax_y=dataset.is_softmax_y)
    surrounding_mcs = 30
    for i in range(0, len(predictions)):
        features = [x for x in dataset.x[i]]
        features.append(dataset.class_to_index(predictions[i]))
        avg_distance_parking_before = 0.0
        avg_distance_parking_after = 0.0
        avg_distance_overtaken_before = 0.0
        avg_distance_overtaken_after = 0.0
        j = i - 1
        while j >= i - surrounding_mcs:
            if j >= 0 and dataset.mcs[j].length_to(dataset.mcs[i]) < 20.0:
                if predictions[j] == 'PARKING_CAR':
                    avg_distance_parking_before = dataset.x[j][0] - dataset.x[i][0]
                    break
                elif predictions[j] == 'OVERTAKING_SITUATION':
                    avg_distance_overtaken_before = dataset.x[j][0] - dataset.x[i][0]
                    break
            j -= 1
        j = i + 1
        while j <= i + surrounding_mcs:
            if j < len(predictions) and dataset.mcs[j].length_to(dataset.mcs[i]) < 20.0:
                if predictions[j] == 'PARKING_CAR':
                    avg_distance_parking_after = dataset.x[j][0] - dataset.x[i][0]
                    break
                elif predictions[j] == 'OVERTAKING_SITUATION':
                    avg_distance_overtaken_after = dataset.x[j][0] - dataset.x[i][0]
                    break
            j += 1
        features.append(avg_distance_parking_before)
        features.append(avg_distance_parking_after)
        features.append(avg_distance_overtaken_before)
        features.append(avg_distance_overtaken_after)
        dataset_normal_plus.x.append(features)
        dataset_normal_plus.y_true.append(dataset_normal.y_true[i])

    return dataset_normal_plus


def enhance_dataset(dataset, predictions, predictions_are_softmax=False):
    dataset_normal_plus = DataSet(class_labels=dataset.class_labels, is_softmax_y=dataset.is_softmax_y)
    surrounding_mcs = 20
    for i in range(0, len(predictions)):
        features = [x for x in dataset.x[i]]
        features.append(dataset.class_to_index(predictions[i]))
        j = i - 1
        while j >= i - surrounding_mcs:
            if j >= 0 and predictions[j] == 'PARKING_CAR' and dataset.mcs[j].length_to(dataset.mcs[i]) < 20.0:
                features.append(dataset.class_to_index(predictions[j]))
                #features.extend(dataset.x[j])
                features.append(dataset.x[j][0])
            else:
                features.append(-1.0)
                features.append(0.0)
                #features.extend([0.0 for cnt in range(len(dataset.x[0]))])
            j -= 1
        j = i + 1
        while j <= i + surrounding_mcs:
            if j < len(predictions) and predictions[j] == 'PARKING_CAR' and dataset.mcs[j].length_to(dataset.mcs[i]) < 20.0:
                features.append(dataset.class_to_index(predictions[j]))
                features.append(dataset.x[j][0])
                #features.extend(dataset.x[j])
            else:
                features.append(-1.0)
                features.append(0.0)
                #features.extend([0.0 for cnt in range(len(dataset.x[0]))])
            j += 1
        dataset_normal_plus.x.append(features)
        dataset_normal_plus.mcs.append(dataset.mcs[i])
        dataset_normal_plus.y_true.append(dataset.y_true[i])

    return dataset_normal_plus


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

    #parking_space_map_clusters, _ = create_parking_space_map(measure_collections_files_dir)
    #measure_collections_files_dir = filter_parking_space_map_mcs(measure_collections_files_dir,
    #                                                             parking_space_map_clusters)

    #print(len(parking_space_map_clusters))
    #print(len(_))

    measure_collections_dir = {}
    for file_name, measure_collections in measure_collections_files_dir.items():
        print(file_name)
        dataset_softmax_10cm = DataSet.get_raw_sensor_dataset_per_10cm(measure_collections, dataset=dataset_softmax_10cm, is_softmax_y=True)
        dataset_normal = DataSet.get_dataset(measure_collections, dataset=dataset_normal)
        measure_collections_dir.update(MeasureCollection.mc_list_to_dict(measure_collections))

    start = time.time()
    # confusion_m_simp = evaluate_model(simple_dense_model, dataset)

    evaluator = DriveByEvaluation()
    # confusion_m_lstm, predictions = evaluator.evaluate(create_random_forest, predict, dataset_normal,
    #                                                   number_of_splits=10, shuffle=True)
    # confusion_m_lstm = evaluator.evaluate(create_random_forest, predict, dataset_normal)
    confusion_m_lstm, predictions = evaluator.evaluate(simple_dense_model, predict_softmax, dataset_softmax_10cm,
                                                       number_of_splits=10, shuffle=True)
    # confusion_m_conv = evaluate_model(create_conv_model, dataset)
    print(time.time() - start)

    # print_confusion_matrix_measures(confusion_m_simp)
    print('lstm')
    print_confusion_matrix_measures(confusion_m_lstm)
    # print('conv')
    # print_confusion_matrix_measures(confusion_m_conv)

    # try double prediction

    dataset_normal_plus = enhance_dataset(dataset_normal, predictions)
    #dataset_normal_plus = enhance_dataset2(dataset_normal, predictions)
    dataset_softmax_plus = enhance_dataset(dataset_softmax_10cm, predictions, predictions_are_softmax=False)

    print('dataset constructed')

    start = time.time()
    evaluator = DriveByEvaluation()
    confusion_m_lstm_after, predictions_after = evaluator.evaluate(create_random_forest,
                                                                   predict,
                                                                   dataset_normal_plus,
                                                                   number_of_splits=10,
                                                                   shuffle=True)
    #confusion_m_lstm_after, predictions_after = evaluator.evaluate(simple_dense_model,
    #                                                               predict_softmax,
    #                                                               dataset_softmax_plus,
    #                                                               number_of_splits=10,
    #                                                               shuffle=True)
    print(time.time() - start)

    cnt_diff = 0
    cnt_neg_to_pos = 0
    cnt_pos_to_neg = 0
    for i in range(len(predictions)):
        if predictions[i] != predictions_after[i]:
            cnt_diff += 1
            print(predictions[i], predictions_after[i])
            if predictions[i] == dataset_normal_plus.y_true[i] and predictions_after[i] != dataset_normal_plus.y_true[i]:
                cnt_pos_to_neg += 1
            elif predictions[i] != dataset_normal_plus.y_true[i] and predictions_after[i] == dataset_normal_plus.y_true[i]:
                cnt_neg_to_pos += 1

    print('cnt diff:', cnt_diff)
    print('pos to neg:', cnt_pos_to_neg)
    print('neg to pos:', cnt_neg_to_pos)

    print_confusion_matrix_measures(confusion_m_lstm_after)
