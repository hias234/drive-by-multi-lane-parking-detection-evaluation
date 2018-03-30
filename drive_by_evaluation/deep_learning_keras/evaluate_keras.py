
from keras.models import Sequential
from keras.layers import Dense, Dropout
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import KFold

from drive_by_evaluation.measure_collection import MeasureCollection
from drive_by_evaluation.db_machine_learning.db_data_set import DataSet

import operator

from drive_by_evaluation.db_machine_learning.confusion_matrix_util import print_confusion_matrix_measures, sumup_confusion_matrices
import time


def dense_5layer32_dropout20_epochs200(dataset, x_train, y_train):
    y_train_softmax = DataSet.to_softmax_y(y_train, dataset.class_labels)

    hidden_dims = 32

    model = Sequential()
    model.add(Dense(hidden_dims, activation='relu', input_dim=len(dataset.x[0])))
    model.add(Dropout(0.2))
    model.add(Dense(hidden_dims, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(hidden_dims, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(hidden_dims, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(hidden_dims, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(len(dataset.class_labels), activation='softmax'))

    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    model.fit(x_train, y_train_softmax,
              epochs=200,
              verbose=0,
              )

    return model


def dense_5layer64_dropout20_epochs200(dataset, x_train, y_train):
    y_train_softmax = DataSet.to_softmax_y(y_train, dataset.class_labels)

    hidden_dims = 64

    model = Sequential()
    model.add(Dense(hidden_dims, activation='relu', input_dim=len(dataset.x[0])))
    model.add(Dropout(0.2))
    model.add(Dense(hidden_dims, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(hidden_dims, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(hidden_dims, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(hidden_dims, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(len(dataset.class_labels), activation='softmax'))

    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    model.fit(x_train, y_train_softmax,
              epochs=200,
              verbose=0,
              )

    return model


def dense_5layer64_dropout20_epochs500(dataset, x_train, y_train):
    y_train_softmax = DataSet.to_softmax_y(y_train, dataset.class_labels)

    hidden_dims = 64

    model = Sequential()
    model.add(Dense(hidden_dims, activation='relu', input_dim=len(dataset.x[0])))
    model.add(Dropout(0.2))
    model.add(Dense(hidden_dims, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(hidden_dims, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(hidden_dims, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(hidden_dims, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(len(dataset.class_labels), activation='softmax'))

    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    model.fit(x_train, y_train_softmax,
              epochs=500,
              verbose=0,
              )

    return model


def dense_5layer32_epochs200(dataset, x_train, y_train):
    y_train_softmax = DataSet.to_softmax_y(y_train, dataset.class_labels)

    hidden_dims = 32

    model = Sequential()
    model.add(Dense(hidden_dims, activation='relu', input_dim=len(dataset.x[0])))
    model.add(Dense(hidden_dims, activation='relu'))
    model.add(Dense(hidden_dims, activation='relu'))
    model.add(Dense(hidden_dims, activation='relu'))
    model.add(Dense(hidden_dims, activation='relu'))
    model.add(Dense(len(dataset.class_labels), activation='softmax'))

    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    model.fit(x_train, y_train_softmax,
              epochs=200,
              verbose=0,
              )

    return model

def dense_5layer64_epochs200(dataset, x_train, y_train):
    y_train_softmax = DataSet.to_softmax_y(y_train, dataset.class_labels)

    hidden_dims = 64

    model = Sequential()
    model.add(Dense(hidden_dims, activation='relu', input_dim=len(dataset.x[0])))
    model.add(Dense(hidden_dims, activation='relu'))
    model.add(Dense(hidden_dims, activation='relu'))
    model.add(Dense(hidden_dims, activation='relu'))
    model.add(Dense(hidden_dims, activation='relu'))
    model.add(Dense(len(dataset.class_labels), activation='softmax'))

    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    model.fit(x_train, y_train_softmax,
              epochs=200,
              verbose=0,
              )

    return model


def dense_5layer64_epochs500(dataset, x_train, y_train):
    y_train_softmax = DataSet.to_softmax_y(y_train, dataset.class_labels)

    hidden_dims = 64

    model = Sequential()
    model.add(Dense(hidden_dims, activation='relu', input_dim=len(dataset.x[0])))
    model.add(Dense(hidden_dims, activation='relu'))
    model.add(Dense(hidden_dims, activation='relu'))
    model.add(Dense(hidden_dims, activation='relu'))
    model.add(Dense(hidden_dims, activation='relu'))
    model.add(Dense(len(dataset.class_labels), activation='softmax'))

    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    model.fit(x_train, y_train_softmax,
              epochs=500,
              verbose=0,
              )

    return model



def evaluate_model(create_model, dataset, n_splits=10, shuffle=True):
    kfold = KFold(n_splits, shuffle)
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


def predict_softmax(model, dataset, x_test, y_test):
    predictions = model.predict(x_test)

    y_pred = []
    #y_true = []

    for i, prediction in enumerate(predictions):
        clazz_predicted, _ = max(enumerate(prediction), key=operator.itemgetter(1))
        class_label_predicted = dataset.class_labels[clazz_predicted]
        #clazz_true, _ = max(enumerate(y_test_softmax[i]), key=operator.itemgetter(1))
        y_pred.append(class_label_predicted)
        #y_true.append(clazz_true)

    return y_pred, y_test


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

    confusion_m_lstm = evaluate_model(dense_5layer64_dropout20_epochs200, dataset)
    #confusion_m_conv = evaluate_model(create_conv_model, dataset)
    print(time.time() - start)

    #print_confusion_matrix_measures(confusion_m_simp)
    print('lstm')
    print_confusion_matrix_measures(confusion_m_lstm)
    #print('conv')
    #print_confusion_matrix_measures(confusion_m_conv)

