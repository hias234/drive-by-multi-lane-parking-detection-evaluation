print(__doc__)

import numpy as np
from scipy import interp
import matplotlib.pyplot as plt
from itertools import cycle

from sklearn import svm, datasets
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import StratifiedKFold


import numpy as np
from drive_by_evaluation.db_machine_learning.multi_scorer import MultiScorer
from drive_by_evaluation.measure_collection import MeasureCollection
from drive_by_evaluation.db_machine_learning.db_data_set import DataSet
from drive_by_evaluation.deep_learning_keras.lstm import create_lstm_model
from drive_by_evaluation.deep_learning_keras.conv import conv_model_128_epochs

import operator

from drive_by_evaluation.db_machine_learning.confusion_matrix_util import print_confusion_matrix_measures, sumup_confusion_matrices
from drive_by_evaluation.parking_map_clustering.dbscan_clustering_directional import create_parking_space_map, filter_parking_space_map_mcs
from drive_by_evaluation.deep_learning_keras.evaluate_keras import dense_5layer64_dropout20_epochs200, predict_softmax
import time

from drive_by_evaluation.db_machine_learning.models import create_random_forest, predict, create_stacked, create_decision_tree, create_mlp

# #############################################################################
# Data IO and generation

# Import some data to play with
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
    dataset_normal = DataSet.get_dataset(measure_collections, dataset=dataset_normal, is_softmax_y=True)
    measure_collections_dir.update(MeasureCollection.mc_list_to_dict(measure_collections))

X = dataset_normal.x
y = dataset_normal.y_true
#X, y = X[y != 2], y[y != 2]
#n_samples, n_features = X.shape

# Add noisy features
random_state = np.random.RandomState(0)
#X = np.c_[X, random_state.randn(n_samples, 200 * n_features)]

# #############################################################################
# Classification and ROC analysis

# Run classifier with cross-validation and plot ROC curves
cv = StratifiedKFold(n_splits=6)
classifier = svm.SVC(kernel='linear', probability=True,
                     random_state=random_state)

tprs = []
aucs = []
mean_fpr = np.linspace(0, 1, 100)

i = 0
for train, test in cv.split(X, y):
    probas_ = classifier.fit(X[train], y[train]).predict_proba(X[test])
    # Compute ROC curve and area the curve
    fpr, tpr, thresholds = roc_curve(y[test], probas_[:, 1])
    tprs.append(interp(mean_fpr, fpr, tpr))
    tprs[-1][0] = 0.0
    roc_auc = auc(fpr, tpr)
    aucs.append(roc_auc)
    plt.plot(fpr, tpr, lw=1, alpha=0.3,
             label='ROC fold %d (AUC = %0.2f)' % (i, roc_auc))

    i += 1
plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
         label='Luck', alpha=.8)

mean_tpr = np.mean(tprs, axis=0)
mean_tpr[-1] = 1.0
mean_auc = auc(mean_fpr, mean_tpr)
std_auc = np.std(aucs)
plt.plot(mean_fpr, mean_tpr, color='b',
         label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),
         lw=2, alpha=.8)

std_tpr = np.std(tprs, axis=0)
tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,
                 label=r'$\pm$ 1 std. dev.')

plt.xlim([-0.05, 1.05])
plt.ylim([-0.05, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic example')
plt.legend(loc="lower right")
plt.show()