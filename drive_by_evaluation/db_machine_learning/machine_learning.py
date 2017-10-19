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


def write_to_file(base_path, ml_file_path):
    files = sorted([f for f in os.listdir(base_path) if os.path.isfile(os.path.join(base_path, f))])

    for f in files:
        data_file = os.path.join(base_path, f)
        camera_folder = os.path.join(base_path, f) + '_images_Camera\\'
        gt_files = [gt_f for gt_f in os.listdir(camera_folder) if gt_f.startswith('00gt')]
        if (len(gt_files) > 0):
            print(gt_files[0])
            measurements1 = Measurement.read(data_file, os.path.join(camera_folder, gt_files[0]))
            measure_collections1 = MeasureCollection.create_measure_collections(measurements1)
            MeasureCollection.write_arff_file(measure_collections1, ml_file_path)


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

    dataset = None
    #write_to_file(base_path, ml_file_path)
    measure_collections_files_dir = MeasureCollection.read_directory(base_path, options=options)

    #parking_space_map_clusters, _ = create_parking_space_map(measure_collections_files_dir)
    #measure_collections_files_dir = filter_parking_space_map_mcs(measure_collections_files_dir, parking_space_map_clusters)

    print(MeasureCollection.get_size(measure_collections_files_dir))

    measure_collections_dir = {}
    for file_name, measure_collections in measure_collections_files_dir.items():
        print(file_name)
        #print(len(measure_collection))
        #measure_collection = filter_acceleration_situations(measure_collection)
        #print('filtered', len(measure_collection))
        #MeasureCollection.write_arff_file(measure_collections1, ml_file_path)
        #measure_collection = [mc for mc in measure_collection if mc.length > 0.5]
        dataset = DataSet.get_dataset(measure_collections, dataset=dataset)
        measure_collections_dir.update(MeasureCollection.mc_list_to_dict(measure_collections))

    classifiers = {
       #'NeuralNetwork': MLPClassifier(),
       #'NeuralNetwork_relu1000': MLPClassifier(activation='relu', max_iter=10000000000),
       #'NeuralNetwork_relu10000_hl5': MLPClassifier(activation='relu', max_iter=100000, hidden_layer_sizes=(50,50,50,50,50)),
       #'NeuralNetwork_relu1000000': MLPClassifier(activation='relu', max_iter=10000000),
       #'DecisionTree_GINI': DecisionTreeClassifier(),
       #'knn20': KNeighborsClassifier(21),
       #'supportVector': SVC(),
       #'gaussian': GaussianProcessClassifier(),
       #'randomforest100': RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=42),
       'randomforest1000': RandomForestClassifier(n_estimators=1000, n_jobs=-1, random_state=42),
       #'randomforest1000_balanced': RandomForestClassifier(n_estimators=1000, n_jobs=-1, random_state=42, class_weight='balanced'),
       #'randomforest10000_balanced': RandomForestClassifier(n_estimators=10000, class_weight='balanced')
       #'custom': SurroundingClf(measure_collections_dir, base_clf=MLPClassifier(), lvl2_clf=MLPClassifier())
    }

    for name, clf in classifiers.items():
        start = time.time()
        scorer = MultiScorer({
            'Accuracy': (accuracy_score, {}),
            'Precision': (precision_score, {'average': 'weighted'}),
            'Recall': (recall_score, {'average': 'weighted'}),
            'ConfusionMatrix': (confusion_matrix, {'labels': dataset.class_labels})
        })
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
        print(time.time() - start)

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
