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
from drive_by_evaluation.ground_truth import GroundTruthClass, GroundTruth
from drive_by_evaluation.measure_collection import MeasureCollection
from drive_by_evaluation.measurement import Measurement

import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.optimizers import SGD
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold

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


def create_keras_model_dense():
    mc = MeasureCollection()
    m = Measurement(1, 1111111111, 48, 14, 4, GroundTruth(1111, GroundTruthClass.FREE_SPACE))
    mc.add_measure(m)
    dataset = DataSet.get_raw_sensor_dataset([mc])
    model = Sequential()
    model.add(Dense(64, activation='relu', input_dim=len(dataset.x[0])))
    model.add(Dropout(0.1))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.1))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.1))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.1))
    model.add(Dense(len(dataset.class_labels), activation='softmax'))

    sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    return model


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

    dataset_raw = None
    dataset_10cm = None
    dataset_10cm_surrounding = None
    dataset_parking = None
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
        dataset_raw = DataSet.get_raw_sensor_dataset(measure_collections, dataset=dataset_raw, is_softmax_y=True)
        dataset_10cm = DataSet.get_raw_sensor_dataset_per_10cm(measure_collections, dataset=dataset_10cm, is_softmax_y=False)
        dataset_10cm_surrounding = DataSet.get_raw_sensor_dataset_per_10cm_p_surroundings(measure_collections, dataset=dataset_10cm_surrounding, is_softmax_y=False)
        dataset_parking = DataSet.get_raw_sensor_dataset_parking_space_detection(measure_collections, dataset=dataset_parking)
        measure_collections_dir.update(MeasureCollection.mc_list_to_dict(measure_collections))

    datasets = {
        #'dataset_raw': dataset_raw,
        'dataset_raw_10cm': dataset_10cm,
        'dataset_raw_10cm_surrounding': dataset_10cm_surrounding,
        #'dataset_raw_parking_space': dataset_parking
    }

    classifiers = {
       #'NeuralNetwork': MLPClassifier(),
       #'NeuralNetwork_relu10000_hl5_10': MLPClassifier(activation='relu', max_iter=100000, hidden_layer_sizes=(100,100,100,100,100)),
       #'NeuralNetwork_relu10000_hl10_50': MLPClassifier(activation='relu', max_iter=100000, hidden_layer_sizes=(50,50,50,50,50,50,50,50,50)),
       'NeuralNetwork_relu10000_hl5': MLPClassifier(activation='relu', max_iter=1000000, hidden_layer_sizes=(50,50,50,50,50)),
        #'Keras': KerasClassifier(build_fn=create_keras_model_dense, epochs=100, batch_size=128, class_weight=dataset_raw.get_class_weights()),
       #'NeuralNetwork_relu100000_hl10': MLPClassifier(activation='relu', max_iter=1000000, hidden_layer_sizes=(100,90,80,70,60,50,40,30,20,10)),
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
