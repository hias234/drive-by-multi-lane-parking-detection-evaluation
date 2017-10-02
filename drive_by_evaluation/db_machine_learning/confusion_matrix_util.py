import numpy as np


def sumup_confusion_matrices(confusion_matrix_list, nr_of_classes):
    confusion_sum = np.zeros((nr_of_classes, nr_of_classes), dtype=int)

    for conf_res in confusion_matrix_list:
        confusion_sum = np.array(conf_res, dtype=int) + confusion_sum

    return confusion_sum


def print_confusion_matrix_measures(confusion_m):
    print(confusion_m)
    true_pos = np.array(np.diag(confusion_m), dtype=float)
    false_pos = np.sum(confusion_m, axis=0) - true_pos
    false_neg = np.sum(confusion_m, axis=1) - true_pos

    print('Accuracy Total: ', np.sum(true_pos) / np.sum(confusion_m))
    print('Recall Total: ', (np.sum(true_pos) / (np.sum(true_pos) + np.sum(false_neg))))
    print('Precision Total: ', (np.sum(true_pos) / (np.sum(true_pos) + np.sum(false_pos))))
    precision = (true_pos / (true_pos + false_pos))
    print('Precision: ', precision)
    recall = (true_pos / (true_pos + false_neg))
    print('Recall: ', recall)
    print('')
