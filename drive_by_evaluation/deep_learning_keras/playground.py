import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Conv1D, MaxPooling1D, GlobalAveragePooling1D
from keras.optimizers import SGD
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold

from drive_by_evaluation.db_machine_learning.multi_scorer import MultiScorer
from drive_by_evaluation.measure_collection import MeasureCollection
from drive_by_evaluation.db_machine_learning.db_data_set import DataSet

print(np.random.random((4, 10)))

print(np.array([1, 2, 3, 4, 5, 6]).reshape(3, 2))

x_train = np.random.random((10, 5, 3))
y_train = keras.utils.to_categorical(np.random.randint(4, size=(100, 1)), num_classes=4)
x_test = np.random.random((20, 5, 3))
y_test = keras.utils.to_categorical(np.random.randint(4, size=(20, 1)), num_classes=4)
#print('x_train[0]', x_train)
#print('y_train[0]', y_train[0])