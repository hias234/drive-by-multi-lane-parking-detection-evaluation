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
    if angle_in_rad > math.pi:
        angle_in_rad = 2 * math.pi - angle_in_rad

    if angle_in_rad > math.pi / 4:
        return 1000

    dist1 = vincenty((mc1[0], mc1[1]), (mc2[0], mc2[1])).meters
    dist2 = vincenty((mc1[2], mc1[3]), (mc2[0], mc2[1])).meters
    dist3 = vincenty((mc1[0], mc1[1]), (mc2[2], mc2[3])).meters
    dist4 = vincenty((mc1[2], mc1[3]), (mc2[2], mc2[3])).meters

    return min([dist1, dist2, dist3, dist4])


# x, y is the point
# p1, p2 are two points of a line
def distance_to_line(x, y, p1_x, p1_y, p2_x, p2_y):
    x_diff = p2_x - p1_x
    y_diff = p2_y - p1_y
    num = abs(y_diff * x - x_diff * y + p2_x * p1_y - p2_y * p1_x)
    den = math.sqrt(y_diff ** 2 + x_diff ** 2)
    return (num / den) / (0.00001 / 1.11)  # convert lat long diff to meters


def create_parking_space_map(measure_collections_files_dir):
    parking_cars_mcs = []
    clustering_data = []
    # measure_collections_dir = {}
    for file_name, measure_collections in measure_collections_files_dir.items():
        parking_cars_mcs.extend(
            [mc for mc in measure_collections if GroundTruthClass.is_parking_car(mc.get_probable_ground_truth())])
        clustering_data.extend([[mc.first_measure().latitude,
                                 mc.first_measure().longitude,
                                 mc.last_measure().latitude,
                                 mc.last_measure().longitude] for mc in measure_collections if
                                GroundTruthClass.is_parking_car(mc.get_probable_ground_truth())])

    print(len(clustering_data))

    db = DBSCAN(metric=mc_metric, eps=8.0, min_samples=2).fit(clustering_data)

    core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
    core_samples_mask[db.core_sample_indices_] = True
    labels = db.labels_

    clusters = []
    noise_cluster = None

    unique_labels = set(labels)

    for k in unique_labels:
        class_member_mask = (labels == k)

        lat = [x[0] for i, x in enumerate(clustering_data) if class_member_mask[i]]
        long = [x[1] for i, x in enumerate(clustering_data) if class_member_mask[i]]
        lat.extend([x[2] for i, x in enumerate(clustering_data) if class_member_mask[i]])
        long.extend([x[3] for i, x in enumerate(clustering_data) if class_member_mask[i]])

        print('k %d' % k)
        print('len %d' % len(lat))

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
                i += 1

            if k != -1 and max_len_indices is not None:
                clusters.append([lat[max_len_indices[0]], long[max_len_indices[0]], lat[max_len_indices[1]], long[max_len_indices[1]]])
            else:
                lat = [x[0] for i, x in enumerate(clustering_data) if class_member_mask[i]]
                long = [x[1] for i, x in enumerate(clustering_data) if class_member_mask[i]]
                noise_cluster = [[lat[i], long[i]] for i in range(0, len(lat))]

    return clusters, noise_cluster


def get_bounding_box(cl):
    m_to_gps = 0.00001 / 1.11
    bounding_distance = 10  # m
    f = bounding_distance * m_to_gps

    dx = cl[2] - cl[0]
    dy = cl[3] - cl[1]
    len_normal_v1 = math.sqrt(dx ** 2 + dy ** 2)
    v1 = [dx / len_normal_v1, dy / len_normal_v1]
    normal_v1 = [-dy / len_normal_v1, dx / len_normal_v1]

    p1 = [cl[0] + (normal_v1[0] - v1[0]) * f, cl[1] + (normal_v1[1] - v1[1]) * f]
    p2 = [cl[0] + (-normal_v1[0] - v1[0]) * f, cl[1] + (-normal_v1[1] - v1[1]) * f]
    p3 = [cl[2] + (normal_v1[0] + v1[0]) * f, cl[3] + (normal_v1[1] + v1[1]) * f]
    p4 = [cl[2] + (-normal_v1[0] + v1[0]) * f, cl[3] + (-normal_v1[1] + v1[1]) * f]

    return [p1, p2, p4, p3]


def is_in_bounding_box(bounding_box, lat, long):
    return is_on_right_of_line(lat, long, bounding_box[0][0], bounding_box[0][1], bounding_box[1][0], bounding_box[1][1]) \
           and is_on_right_of_line(lat, long, bounding_box[1][0], bounding_box[1][1], bounding_box[2][0], bounding_box[2][1]) \
           and is_on_right_of_line(lat, long, bounding_box[2][0], bounding_box[2][1], bounding_box[3][0], bounding_box[3][1]) \
           and is_on_right_of_line(lat, long, bounding_box[3][0], bounding_box[3][1], bounding_box[0][0], bounding_box[0][1])


def is_on_right_of_line(x, y, p1_x, p1_y, p2_x, p2_y):
    d = (x - p1_x) * (p2_y - p1_y) - (y - p1_y) * (p2_x - p1_x)
    return d < 0


def filter_parking_space_map_mcs(mc_map, clusters):
    new_mc_map = {}
    for name, mc_list in mc_map.items():
        new_mc_list = []
        for mc in mc_list:
            for cl in clusters:
                bounding_box = get_bounding_box(cl)


                # len_between1 = vincenty((mc.first_measure().latitude, mc.first_measure().longitude), (cl[0], cl[1])).meters
                # len_between2 = vincenty((mc.last_measure().latitude, mc.last_measure().longitude), (cl[0], cl[1])).meters
                # len_between3 = vincenty((mc.first_measure().latitude, mc.first_measure().longitude), (cl[2], cl[3])).meters
                # len_between4 = vincenty((mc.last_measure().latitude, mc.last_measure().longitude), (cl[2], cl[3])).meters
                #
                # #len_between5 = distance_to_line(mc.first_measure().latitude, mc.first_measure().longitude, cl[0], cl[1], cl[2], cl[3])
                # #len_between6 = distance_to_line(mc.last_measure().latitude, mc.last_measure().longitude, cl[0], cl[1], cl[2], cl[3])
                #
                # if min(len_between1, len_between2, len_between3, len_between4) < 10.0:
                if is_in_bounding_box(bounding_box, mc.first_measure().latitude, mc.first_measure().longitude)\
                        or is_in_bounding_box(bounding_box, mc.last_measure().latitude, mc.last_measure().longitude):
                    new_mc_list.append(mc)
                    break

        new_mc_map[name] = new_mc_list
    return new_mc_map


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

    # dataset = None
    measure_collections_files_dir = MeasureCollection.read_directory(base_path, options=options)

    clusters, noise_cluster = create_parking_space_map(measure_collections_files_dir)

    gmap = gmplot.GoogleMapPlotter(48.3045, 14.291153333, 16)

    for k in range(0, len(clusters)):
        gmap.plot([clusters[k][0], clusters[k][2]],
                  [clusters[k][1], clusters[k][3]], edge_width=5)

        bounding_box = get_bounding_box(clusters[k])
        gmap.polygon([p[0] for p in bounding_box], [p[1] for p in bounding_box])

    print('mcs ', [len(mc_list) for name, mc_list in measure_collections_files_dir.items()])
    filtered_mcs = filter_parking_space_map_mcs(measure_collections_files_dir, clusters)
    print('filtered mcs ', [len(mc_list) for name, mc_list in filtered_mcs.items()])

    for name, mc_list in filtered_mcs.items():
        gmap.scatter(
            [mc.center_latitude for mc in mc_list if GroundTruthClass.is_parking_car(mc.get_probable_ground_truth())],
            [mc.center_longitude for mc in mc_list if GroundTruthClass.is_parking_car(mc.get_probable_ground_truth())],
            '#00FF00', size=2, marker=False)
        gmap.scatter([mc.center_latitude for mc in mc_list if
                      not GroundTruthClass.is_parking_car(mc.get_probable_ground_truth())],
                     [mc.center_longitude for mc in mc_list if
                      not GroundTruthClass.is_parking_car(mc.get_probable_ground_truth())],
                     '#000000', size=1, marker=False)

    if noise_cluster is not None:
        lat = [noise[0] for noise in noise_cluster]
        long = [noise[1] for noise in noise_cluster]
        gmap.scatter(lat, long, '#0000FF', size=4, marker=False)

    gmap.draw("C:\\sw\\master\\mymap_parking_clusters_directional.html")
