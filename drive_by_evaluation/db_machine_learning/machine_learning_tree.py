
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import KFold

from drive_by_evaluation.db_machine_learning.db_data_set import DataSet
from drive_by_evaluation.db_machine_learning.multi_scorer import MultiScorer
from drive_by_evaluation.measure_collection import MeasureCollection
from sklearn import tree

from drive_by_evaluation.parking_map_clustering.dbscan_clustering_directional import create_parking_space_map, filter_parking_space_map_mcs

if __name__ == '__main__':
    base_path = 'C:\\sw\\master\\collected data\\'
    #base_path = 'C:\\sw\\master\\collected data\\data_20170718_tunnel\\'

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

    parking_space_map_clusters, _ = create_parking_space_map(measure_collections_files_dir)
    measure_collections_files_dir = filter_parking_space_map_mcs(measure_collections_files_dir, parking_space_map_clusters)

    for file_name, measure_collections in measure_collections_files_dir.items():
        print(file_name)
        dataset = DataSet.get_dataset(measure_collections, dataset=dataset)
        measure_collections_dir.update(MeasureCollection.mc_list_to_dict(measure_collections))

    dataset.to_arff_file('parking_map_filtered_dataset.arff')

    classifiers = {
       'DecisionTree_GINI': DecisionTreeClassifier(max_depth=3),
    }

    for name, clf in classifiers.items():
        clf.fit(dataset.x, dataset.y_true)

        import pydot
        from io import StringIO

        #dot_data = StringIO()
        tree.export_graphviz(clf, out_file='tree_pruned_parknet.dot')
        #print(dot_data.getvalue())
        #graph = pydot.graph_from_dot_data(dot_data.getvalue())
        #print(graph)
        #graph[0].write_pdf("iris.pdf")

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



