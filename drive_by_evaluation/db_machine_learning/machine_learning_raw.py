import os
import random

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


def get_dataset_parking_cars(measure_collections, dataset=None):
    if dataset is None:
        dataset = DataSet(['PARKING_CAR', 'NO_PARKING_CAR'])

    for mc in measure_collections:
        features = [mc.avg_distance, mc.get_length(), mc.get_duration(), mc.get_nr_of_measures(),
                    mc.get_distance_variance(), mc.avg_speed, mc.get_acceleration(),
                    mc.first_measure().distance, mc.measures[len(mc.measures) / 2].distance, mc.last_measure().distance
                    ]

        for interval, surrounding_mc in mc.time_surrounding_mcs.iteritems():
            features.append(surrounding_mc.avg_distance)
            features.append(surrounding_mc.avg_speed)
            features.append(surrounding_mc.length)
            features.append(surrounding_mc.get_acceleration())

        ground_truth = 'NO_PARKING_CAR'
        gt = mc.get_probable_ground_truth()
        if GroundTruthClass.is_parking_car(gt):
            ground_truth = 'PARKING_CAR'

        dataset.append_sample(features, ground_truth)

    return dataset


def get_overtaking_situation_dataset(measure_collections, dataset=None):
    if dataset is None:
        dataset = DataSet(['NO_OVERTAKING_SITUATION', 'OVERTAKING_SITUATION'])

    for mc in measure_collections:
        if mc.get_length() > 1.0:
            features = [mc.avg_distance, mc.get_length(), mc.get_duration(), mc.get_nr_of_measures(),
                        mc.get_distance_variance(), mc.avg_speed, mc.get_acceleration(),
                        mc.first_measure().distance, mc.measures[len(mc.measures) / 2].distance,
                        mc.last_measure().distance]

            for interval, surrounding_mc in mc.time_surrounding_mcs.iteritems():
                features.append(surrounding_mc.avg_distance)
                features.append(surrounding_mc.avg_speed)
                features.append(surrounding_mc.length)
                features.append(surrounding_mc.get_acceleration())

            ground_truth = 'NO_OVERTAKING_SITUATION'
            gt = mc.get_probable_ground_truth()
            if GroundTruthClass.is_overtaking_situation(gt):
                ground_truth = 'OVERTAKING_SITUATION'

            # undersampling
            if not GroundTruthClass.is_overtaking_situation(gt) and random.randint(0, 10) < 10:
                dataset.append_sample(features, ground_truth)
            elif GroundTruthClass.is_overtaking_situation(gt):
                i = 0
                while i < 3:
                    dataset.append_sample(features, ground_truth)
                    i += 1

    return dataset


def filter_acceleration_situations(measure_collections):
    i = 0
    for measure_collection in measure_collections:
        #print(measure_collection.time_surrounding_mcs.get(10.0).get_acceleration())
        if measure_collection.time_surrounding_mcs.get(10.0).get_acceleration() < -0.2:
            measure_collections.pop(i)
            #print(measure_collection.time_surrounding_mcs.get(10.0).get_acceleration())
        else:
            i += 1

    return measure_collections


if __name__ == '__main__':
    base_path = 'C:\\sw\\master\\collected data\\'
    #base_path = 'C:\\sw\\master\\collected data\\data_20170718_tunnel\\'
    # ml_file_path = 'C:\\sw\\master\\00ml.arff'
    ml_file_path = 'C:\\sw\\master\\20170718ml.arff'

    options = {
        'mc_min_speed': 1.0,
        'mc_merge': True,
        'mc_separation_threshold': 1.0,
        'mc_min_measure_count': 2,
        # 'mc_surrounding_times_s': [2.0, 5.0],
        'outlier_threshold_distance': 1.0,
        'outlier_threshold_diff': 0.5,
        # 'replacement_values': {0.01: 10.01},
        'min_measurement_value': 0.06,
    }

    dataset_raw = None
    dataset_10cm = None
    #write_to_file(base_path, ml_file_path)
    measure_collections_files_dir = MeasureCollection.read_directory(base_path, options=options)
    measure_collections_dir = {}
    for file_name, measure_collections in measure_collections_files_dir.items():
        print(file_name)
        #print(len(measure_collection))
        #measure_collection = filter_acceleration_situations(measure_collection)
        #print('filtered', len(measure_collection))
        #MeasureCollection.write_arff_file(measure_collections1, ml_file_path)
        #measure_collection = [mc for mc in measure_collection if mc.length > 0.5]
        dataset_raw = DataSet.get_raw_sensor_dataset(measure_collections, dataset=dataset_raw)
        dataset_10cm = DataSet.get_raw_sensor_dataset_per_10cm(measure_collections, dataset=dataset_10cm)
        measure_collections_dir.update(MeasureCollection.mc_list_to_dict(measure_collections))

    datasets = {
        'dataset_raw': dataset_raw,
        'dataset_raw_10cm': dataset_10cm
    }

    classifiers = {
       'NeuralNetwork_relu10000_hl5': MLPClassifier(activation='relu', max_iter=100000, hidden_layer_sizes=(50,50,50,50,50)),
       'NeuralNetwork_relu100000_hl10': MLPClassifier(activation='relu', max_iter=1000000, hidden_layer_sizes=(100,90,80,70,60,50,40,30,20,10)),
    }

    for dataset_name, dataset in datasets.items():
        for name, clf in classifiers.items():
            scorer = MultiScorer({
                'Accuracy': (accuracy_score, {}),
                'Precision': (precision_score, {'average': 'weighted'}),
                'Recall': (recall_score, {'average': 'weighted'}),
                'ConfusionMatrix': (confusion_matrix, {'labels': dataset.class_labels})
            })
            print(dataset_name)
            print(name)

            # X_train, X_test, y_train, y_test = train_test_split(dataset.x, dataset.y_true, test_size=0.33, random_state=42)
            # clf.fit(X_train, y_train)
            # print('fitted')
            # i = 0
            # mismatches = []
            # while i < len(X_test[0]):
            #      predicted = clf.predict(np.array(X_test), [[1, 15], [15, 0]])
            #                              #.reshape(1, -1))
            #      #print(predicted[0])
            #      #print(dataset_test[1][i])
            #      if predicted[0] != y_test[i]:
            #           print('features: ', X_test)
            #           print('GroundTruth: ', y_test)
            #           print('Predicted: ', predicted[0])
            #           print('')
            #           mismatches.append((X_test, y_test, predicted[0]))
            #      i += 1
            # print(len(mismatches))
            #
            # continue
            kfold = KFold(n_splits=5)
            cross_val_score(clf, dataset.x, dataset.y_true, cv=kfold, scoring=scorer)
            results = scorer.get_results()

            confusion_m = None
            for metric_name in results.keys():
                if metric_name == 'ConfusionMatrix':
                    print(metric_name)
                    confusion_m = np.sum(results[metric_name], axis=0)
                    print(dataset.class_labels)
                    print(confusion_m)
                else:
                    print(metric_name, np.average(results[metric_name]))

            true_pos = np.array(np.diag(confusion_m), dtype=float)
            false_pos = np.sum(confusion_m, axis=0) - true_pos
            false_neg = np.sum(confusion_m, axis=1) - true_pos

            precision = (true_pos / (true_pos + false_pos))
            print('Precision: ', precision)
            recall = (true_pos / (true_pos + false_neg))
            print('Recall: ', recall)
            print('')
