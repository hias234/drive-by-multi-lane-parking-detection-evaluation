from sklearn.base import BaseEstimator, ClassifierMixin
import numpy as np


class SurroundingClf(BaseEstimator, ClassifierMixin):
    """An example of classifier"""

    def __init__(self, measure_collection_dir, base_clf=None, lvl2_clf=None):
        self.measure_collection_dir = measure_collection_dir
        print measure_collection_dir
        self.base_clf = base_clf
        self.lvl2_clf = lvl2_clf
        self.extended_x = None

    def fit(self, X, y):
        self.base_clf.fit(X, y)

        self.extended_x = []

        # dict where key = id and value = predicted
        y_pred = {x[0]: self.base_clf.predict(np.array(x).reshape(1, -1))[0] for x in X}
        print y_pred

        for index, x in enumerate(X):
            id = x[0]
            mc = self.measure_collection_dir[id]
            cur_id = id - 1

            cnt_mcs = 0
            cnt_parking_mcs = 0
            sum_distance_parking_mcs = 0

            while cur_id in self.measure_collection_dir and cur_id in y_pred:
                cur_mc = self.measure_collection_dir[cur_id]
                if (abs(mc.first_measure().timestamp - cur_mc.first_measure().timestamp) < 10
                    and mc.length_to(cur_mc) < 10):
                    cnt_mcs += 1
                    if y_pred[cur_id] == 'PARKING_CAR':
                        cnt_parking_mcs += 1
                        sum_distance_parking_mcs += cur_mc.avg_distance
                else:
                    break
                cur_id -= 1

            cur_extended_x = x
            cur_extended_x.extend([0 if y_pred[id] == 'PARKING_CAR' else 1, cnt_mcs, cnt_parking_mcs, 100000 if cnt_parking_mcs == 0 else sum_distance_parking_mcs / cnt_parking_mcs])
            self.extended_x.append(cur_extended_x)

        self.lvl2_clf.fit(self.extended_x, y)

        return self.lvl2_clf

    def predict(self, X):
        

        return self.lvl2_clf.predict(X)

    def score(self, X, y):
        return 0
