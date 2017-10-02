
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

import operator

from drive_by_evaluation.db_machine_learning.confusion_matrix_util import print_confusion_matrix_measures, sumup_confusion_matrices

#base_path = 'C:\\sw\\master\\collected data\\data_20170725_linz\\'
base_path = 'C:\\sw\\master\\collected data\\'

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

#def evaluate_model(create_model)

dataset = None
measure_collections_files_dir = MeasureCollection.read_directory(base_path, options=options)
measure_collections_dir = {}
for file_name, measure_collections in measure_collections_files_dir.items():
    print(file_name)
    dataset = DataSet.get_raw_sensor_dataset_per_10cm(measure_collections, dataset=dataset, is_softmax_y=True)
    measure_collections_dir.update(MeasureCollection.mc_list_to_dict(measure_collections))

kfold = KFold(n_splits=5, shuffle=False)
confusion_res = []
for train, test in kfold.split(dataset.x, dataset.y_true):
    #print(dataset.x)
    print(train)
    train0 = train[0]
    trainlast = train[len(train) - 1]
    test0 = test[0]
    testlast = test[len(test) - 1]
    print(trainlast - train0)
    #print(dataset.x[train][0])
    #print(dataset.y_true[train][0])

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

    model.fit(dataset.x[train0:trainlast], dataset.y_true[train0:trainlast],
              epochs=100,
              class_weight=dataset.get_class_weights()
              )

    predictions = model.predict(dataset.x[test0:testlast])

    y_pred = []
    y_true = []

    i = test0
    for prediction in predictions:
        clazz_predicted, _ = max(enumerate(prediction), key=operator.itemgetter(1))
        clazz_true, _ = max(enumerate(dataset.y_true[i]), key=operator.itemgetter(1))
        y_pred.append(clazz_predicted)
        y_true.append(clazz_true)
        i += 1

    confusion_m = confusion_matrix(y_true, y_pred, labels=range(0, len(dataset.class_labels)))
    print(confusion_m)
    print_confusion_matrix_measures(confusion_m)
    confusion_res.append(confusion_m)

confusion_sum = sumup_confusion_matrices(confusion_res, dataset.get_nr_of_classes())
print_confusion_matrix_measures(confusion_sum)
