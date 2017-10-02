
import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, LSTM, Embedding
from keras.optimizers import SGD
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold

from drive_by_evaluation.db_machine_learning.multi_scorer import MultiScorer
from drive_by_evaluation.measure_collection import MeasureCollection
from drive_by_evaluation.db_machine_learning.db_data_set import DataSet


base_path = 'C:\\sw\\master\\collected data\\data_20170725_linz\\'
#base_path = 'C:\\sw\\master\\collected data\\'

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

dataset = None
measure_collections_files_dir = MeasureCollection.read_directory(base_path, options=options)
measure_collections_dir = {}
for file_name, measure_collections in measure_collections_files_dir.items():
    print(file_name)
    dataset = DataSet.get_raw_sensor_dataset_per_10cm(measure_collections, dataset=dataset, use_floats=True)
    measure_collections_dir.update(MeasureCollection.mc_list_to_dict(measure_collections))

# Generate dummy data
x_train = [x_t for i, x_t in enumerate(dataset.x) if i < len(dataset.x) * 0.8]
y_train = keras.utils.to_categorical([x_t for i, x_t in enumerate(dataset.y_true) if i < len(dataset.x) * 0.8],
                                     num_classes=len(dataset.class_labels))
x_test = [x_t for i, x_t in enumerate(dataset.x) if i >= len(dataset.x) * 0.8]
y_test = keras.utils.to_categorical([x_t for i, x_t in enumerate(dataset.y_true) if i >= len(dataset.x) * 0.8],
                                    num_classes=len(dataset.class_labels))

print('x_train[0]', x_train[0])
print('y_train[0]', y_train[0])

max_features = 128
hidden_dims = 64

model = Sequential()
model.add(Embedding(max_features, output_dim=256))
model.add(LSTM(128))
model.add(Dropout(0.2))
model.add(Dense(hidden_dims, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(len(dataset.class_labels), activation='softmax'))

sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

model.fit(x_train, y_train,
          epochs=20)
score = model.evaluate(x_test, y_test)

print('score', score)
