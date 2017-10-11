import numpy as np

from drive_by_evaluation.ground_truth import GroundTruthClass
from drive_by_evaluation.measure_collection import MeasureCollection
from sklearn.cluster import DBSCAN
from sklearn import metrics
from geopy.distance import vincenty

import matplotlib.pyplot as plt
import gmplot

import pprint
import math

pp = pprint.PrettyPrinter(indent=4)


def unit_vector(vector):
    """ Returns the unit vector of the vector.  """
    return vector / np.linalg.norm(vector)


def angle_between(v1, v2):
    """ Returns the angle in radians between vectors 'v1' and 'v2'::

            >>> angle_between((1, 0, 0), (0, 1, 0))
            1.5707963267948966
            >>> angle_between((1, 0, 0), (1, 0, 0))
            0.0
            >>> angle_between((1, 0, 0), (-1, 0, 0))
            3.141592653589793
    """
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))


def mc_metric(mc1, mc2):
    # [mc.first_measure().latitude, mc.first_measure().longitude, mc.last_measure().latitude, mc.last_measure().longitude]
    direction1 = (mc1[0] - mc1[2], mc1[1] - mc1[3])
    direction2 = (mc2[0] - mc2[2], mc2[1] - mc2[3])
    angle_in_rad = angle_between(direction1, direction2)

    if abs(angle_in_rad - math.pi / 2) < math.pi / 4:
        return 1000

    dist1 = vincenty((mc1[0], mc1[1]), (mc2[0], mc2[1])).meters
    dist2 = vincenty((mc1[2], mc1[3]), (mc2[0], mc2[1])).meters
    dist3 = vincenty((mc1[0], mc1[1]), (mc2[2], mc2[3])).meters
    dist4 = vincenty((mc1[2], mc1[3]), (mc2[2], mc2[3])).meters

    return min([dist1, dist2, dist3, dist4])


if __name__ == '__main__':
    base_path = 'C:\\sw\\master\\collected data\\data_20170718_tunnel\\'
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

    #dataset = None
    measure_collections_files_dir = MeasureCollection.read_directory(base_path, options=options)

    parking_cars_mcs = []
    clustering_data = []
    #measure_collections_dir = {}
    for file_name, measure_collections in measure_collections_files_dir.items():
        parking_cars_mcs.extend([mc for mc in measure_collections if GroundTruthClass.is_parking_car(mc.get_probable_ground_truth())])
        clustering_data.extend([[mc.first_measure().latitude, mc.first_measure().longitude, mc.last_measure().latitude, mc.last_measure().longitude] for mc in measure_collections if GroundTruthClass.is_parking_car(mc.get_probable_ground_truth())])
        #print(file_name)
        #dataset = DataSet.get_raw_sensor_dataset_per_10cm(measure_collections, dataset=dataset, is_softmax_y=True)
        #measure_collections_dir.update(MeasureCollection.mc_list_to_dict(measure_collections))

    print(len(clustering_data))

    db = DBSCAN(metric=mc_metric, eps=3.0, min_samples=3).fit(clustering_data)
    core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
    core_samples_mask[db.core_sample_indices_] = True
    labels = db.labels_

    # Number of clusters in labels, ignoring noise if present.
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)

    print('Estimated number of clusters: %d' % n_clusters_)
    #print("Homogeneity: %0.3f" % metrics.homogeneity_score(labels_true, labels))
    #print("Completeness: %0.3f" % metrics.completeness_score(labels_true, labels))
    #print("V-measure: %0.3f" % metrics.v_measure_score(labels_true, labels))
    #print("Adjusted Rand Index: %0.3f"
    #      % metrics.adjusted_rand_score(labels_true, labels))
    #print("Adjusted Mutual Information: %0.3f"
    #      % metrics.adjusted_mutual_info_score(labels_true, labels))
    #print("Silhouette Coefficient: %0.3f"
    #      % metrics.silhouette_score(X, labels))

    unique_labels = set(labels)
    colors = [plt.cm.Spectral(each) for each in np.linspace(0, 1, len(unique_labels))]

    gmap = gmplot.GoogleMapPlotter(48.3045, 14.291153333, 16)

    print('len points: %d' % len(clustering_data))
    for k, col in zip(unique_labels, colors):
        if k == -1:
            # Black used for noise.
            col = [0, 0, 0, 1]

        class_member_mask = (labels == k)
        #print(class_member_mask)
        #print(core_samples_mask)

        lat = [x[0] for i, x in enumerate(clustering_data) if class_member_mask[i]] #if class_member_mask[i] and core_samples_mask[i]
        long = [x[1] for i, x in enumerate(clustering_data) if class_member_mask[i]]

        #plt.plot(lat, long, 'o', markerfacecolor=tuple(col),
        #         markeredgecolor='k', markersize=14)
        print('k %d' % k)
        print('len %d' % len(lat))
        #pp.pprint(lat)
        #pp.pprint(long)

        if len(lat) > 0 and len(long) > 0:
            i = 0
            max_len_between = 0.0
            max_len_indices = None
            while i < len(lat) - 1:
                j = i + 1
                while j < len(lat):
                    len_between = vincenty((lat[i], long[i]), (lat[j], long[j])).meters
                    if max_len_indices is None or len_between > max_len_between:
                        max_len_between = len_between
                        max_len_indices = [i, j]
                    j += 1
                    #print(str(i) + ' ' + str(j))
                i += 1

            if k != -1 and max_len_indices is not None:
                #gmap.scatter(lat, long, '#00FF00', size=2, marker=False)
                print('maxlen: %d m' % max_len_between)
                gmap.plot([lat[max_len_indices[0]], lat[max_len_indices[1]]],
                          [long[max_len_indices[0]], long[max_len_indices[1]]], edge_width=5) #, edge_color="cyan", edge_width=5)
            elif k == -1:
                gmap.scatter(lat, long, '#0000FF', size=4, marker=False)

        #lat = [x[0] for i, x in enumerate(clustering_x) if class_member_mask[i] and not core_samples_mask[i]]
        #long = [x[1] for i, x in enumerate(clustering_x) if class_member_mask[i] and not core_samples_mask[i]]
        #plt.plot(lat, long, 'o', markerfacecolor=tuple(col),
        #         markeredgecolor='k', markersize=6)

    #plt.title('Estimated number of clusters: %d' % n_clusters_)
    #plt.show()

    gmap.draw("C:\\sw\\master\\mymap_parking_clusters_directional.html")
