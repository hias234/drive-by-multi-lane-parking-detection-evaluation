
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
import time


def simple_dense_model(dataset, x_train, y_train):
    model = Sequential()
    model.add(Dense(64, activation='relu', input_dim=len(dataset.x[0])))
    model.add(Dropout(0.1))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.1))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.1))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.1))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.1))
    model.add(Dense(len(dataset.class_labels), activation='softmax'))

    sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    model.fit(x_train, y_train,
              epochs=200,
              #class_weight=dataset.get_class_weights()
              )

    return model


def evaluate_model(create_model, dataset):
    kfold = KFold(n_splits=5, shuffle=False)
    confusion_res = []
    for train, test in kfold.split(dataset.x, dataset.y_true):
        x_train = [x for i, x in enumerate(dataset.x) if i in train]
        y_train = [x for i, x in enumerate(dataset.y_true) if i in train]
        x_test = [x for i, x in enumerate(dataset.x) if i in test]
        y_test = [x for i, x in enumerate(dataset.y_true) if i in test]

        model = create_model(dataset, x_train, y_train)

        y_pred, y_true = predict_softmax(model, x_test, y_test)

        confusion_m = confusion_matrix(y_true, y_pred, labels=range(0, len(dataset.class_labels)))
        print_confusion_matrix_measures(confusion_m)
        confusion_res.append(confusion_m)

    confusion_sum = sumup_confusion_matrices(confusion_res, dataset.get_nr_of_classes())
    print_confusion_matrix_measures(confusion_sum)

    return confusion_sum


def predict_softmax(model, x_test, y_test):
    predictions = model.predict(x_test)

    y_pred = []
    y_true = []

    for i, prediction in enumerate(predictions):
        clazz_predicted, _ = max(enumerate(prediction), key=operator.itemgetter(1))
        clazz_true, _ = max(enumerate(y_test[i]), key=operator.itemgetter(1))
        y_pred.append(clazz_predicted)
        y_true.append(clazz_true)

    return y_pred, y_true


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

    dataset = None
    measure_collections_files_dir = MeasureCollection.read_directory(base_path, options=options)

    #parking_space_map_clusters, _ = create_parking_space_map(measure_collections_files_dir)
    #measure_collections_files_dir = filter_parking_space_map_mcs(measure_collections_files_dir,
    #                                                             parking_space_map_clusters)

    measure_collections_dir = {}
    for file_name, measure_collections in measure_collections_files_dir.items():
        print(file_name)
        # dataset = DataSet.get_raw_sensor_dataset_per_10cm(measure_collections, dataset=dataset, is_softmax_y=True)
        dataset = DataSet.get_raw_sensor_dataset_per_10cm_p_surroundings(measure_collections, dataset=dataset, is_softmax_y=True)
        measure_collections_dir.update(MeasureCollection.mc_list_to_dict(measure_collections))

    start = time.time()
    #confusion_m_simp = evaluate_model(simple_dense_model, dataset)

    confusion_m_lstm = evaluate_model(simple_dense_model, dataset)
    #confusion_m_conv = evaluate_model(create_conv_model, dataset)
    print(time.time() - start)

    #print_confusion_matrix_measures(confusion_m_simp)
    print('lstm')
    print_confusion_matrix_measures(confusion_m_lstm)
    #print('conv')
    #print_confusion_matrix_measures(confusion_m_conv)

