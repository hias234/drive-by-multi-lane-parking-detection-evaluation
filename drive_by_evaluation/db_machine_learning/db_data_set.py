from drive_by_evaluation.ground_truth import GroundTruthClass
from drive_by_evaluation.measure_collection import MeasureCollection
from drive_by_evaluation.parking_map_clustering.dbscan_clustering_directional import create_parking_space_map, filter_parking_space_map_mcs
import numpy as np
import keras
import os


class DataSet:
    def __init__(self, name, class_labels, is_softmax_y):
        self.name = name
        self.x = []
        self.y_true = []
        self.mcs = []
        self.class_labels = class_labels
        self.is_softmax_y = is_softmax_y

    def append_sample(self, x, y_true, mc):
        self.x.append(x)
        self.y_true.append(y_true)
        self.mcs.append(mc)

    def get_nr_of_classes(self):
        return len(self.class_labels)

    def class_to_index(self, clazz):
        return self.class_labels.index(clazz)

    def get_class_weights(self):
        class_weights = {}
        for y in self.y_true:
            if self.is_softmax_y:
                for i, y_i in enumerate(y):
                    class_weights[i] = class_weights.get(i, len(self.x)) - y_i
            else:
                class_weights[y] = class_weights.get(y, len(self.x)) - 1
        return class_weights

    @staticmethod
    def get_four_classes_groundtruth(mc):
        ground_truth = 'FREE_SPACE'
        gt = mc.get_probable_ground_truth()
        if GroundTruthClass.is_parking_car(gt):
            ground_truth = 'PARKING_CAR'
        elif GroundTruthClass.is_overtaking_situation(gt):
            ground_truth = 'OVERTAKING_SITUATION'
        elif GroundTruthClass.is_parking_motorcycle_or_bicycle(gt):
            ground_truth = 'PARKING_MC_BC'
        return ground_truth

    @staticmethod
    def get_parking_classes_groundtruth(mc):
        ground_truth = 'NO_PARKING_CAR'
        gt = mc.get_probable_ground_truth()
        if GroundTruthClass.is_parking_car(gt):
            ground_truth = 'PARKING_CAR'
        return ground_truth

    @staticmethod
    def append_to_dataset(dataset, features, mc, ground_truth, class_labels):
        # if dataset.is_softmax_y:
        #     index = class_labels.index(ground_truth)
        #     softmax_y = keras.utils.to_categorical(index, num_classes=len(class_labels))[0].tolist()
        #     dataset.append_sample(features, softmax_y, mc)
        # else:
        dataset.append_sample(features, ground_truth, mc)
        return dataset

    @staticmethod
    def to_softmax_y(ys, class_labels):
        ys_softmax = []
        for y in ys:
            index = class_labels.index(y)
            softmax_y = keras.utils.to_categorical(index, num_classes=len(class_labels))[0].tolist()
            ys_softmax.append(softmax_y)
        return ys_softmax

    @staticmethod
    def get_2d_dataset(measure_collections, dataset=None, is_softmax_y=False):
        class_labels = ['FREE_SPACE', 'PARKING_CAR', 'OVERTAKING_SITUATION', 'PARKING_MC_BC']
        if dataset is None:
            dataset = DataSet('2d_full_dataset', class_labels, is_softmax_y)

        for i, mc in enumerate(measure_collections):
            last_distance = 0 if i == 0 else measure_collections[i - 1].avg_distance  # .last_measure().distance
            next_distance = 0 if len(measure_collections) == i + 1 else measure_collections[
                i + 1].avg_distance  # .first_measure().distance
            features = [  # mc.id,
                mc.avg_distance,
                mc.get_length()
            ]

            ground_truth = DataSet.get_four_classes_groundtruth(mc)
            dataset = DataSet.append_to_dataset(dataset, features, mc, ground_truth, class_labels)

        return dataset

    @staticmethod
    def get_dataset(measure_collections, dataset=None, is_softmax_y=False, name='full_dataset'):
        class_labels = ['FREE_SPACE', 'PARKING_CAR', 'OVERTAKING_SITUATION', 'PARKING_MC_BC']
        if dataset is None:
            dataset = DataSet(name, class_labels, is_softmax_y)

        for i, mc in enumerate(measure_collections):
            last_distance = 0 if i == 0 else measure_collections[i - 1].avg_distance  # .last_measure().distance
            next_distance = 0 if len(measure_collections) == i + 1 else measure_collections[
                i + 1].avg_distance  # .first_measure().distance
            features = [#mc.id,
                        mc.avg_distance,
                        mc.get_length(),
                        mc.get_duration(),
                        mc.get_nr_of_measures(),
                        mc.get_distance_variance(),
                        mc.avg_speed,
                        mc.get_acceleration(),
                        # last_distance,
                        # next_distance,
                        mc.avg_distance - last_distance,
                        mc.avg_distance - next_distance,
                        #mc.first_measure().distance - last_distance,
                        #mc.last_measure().distance - next_distance,
                        #mc.first_measure().distance,
                        #mc.measures[int(len(mc.measures) / 2)].distance,
                        #mc.measures[int(len(mc.measures) / 4)].distance,
                        #mc.measures[int(len(mc.measures) / 4 * 3)].distance,
                        #mc.last_measure().distance
                        ]

            for interval, surrounding_mc in mc.time_surrounding_mcs.items():
                features.append(surrounding_mc.avg_distance)
                features.append(surrounding_mc.avg_speed)
                features.append(surrounding_mc.length)
                features.append(surrounding_mc.get_acceleration())

            ground_truth = DataSet.get_four_classes_groundtruth(mc)
            dataset = DataSet.append_to_dataset(dataset, features, mc, ground_truth, class_labels)

        return dataset

    @staticmethod
    def get_filtered_feature_dataset(measure_collections, dataset=None, is_softmax_y=False, name='less_features_dataset'):
        class_labels = ['FREE_SPACE', 'PARKING_CAR', 'OVERTAKING_SITUATION', 'PARKING_MC_BC']
        if dataset is None:
            dataset = DataSet(name, class_labels, is_softmax_y)

        for i, mc in enumerate(measure_collections):
            last_distance = 0 if i == 0 else measure_collections[i - 1].avg_distance  # .last_measure().distance
            next_distance = 0 if len(measure_collections) == i + 1 else measure_collections[
                i + 1].avg_distance  # .first_measure().distance
            features = [
                mc.avg_distance,
                mc.get_length(),
                mc.get_duration(),
                mc.get_nr_of_measures(),
                mc.avg_distance - last_distance,
                mc.avg_distance - next_distance,
            ]

            for interval, surrounding_mc in mc.time_surrounding_mcs.items():
                features.append(surrounding_mc.avg_distance)
                features.append(surrounding_mc.avg_speed)
                features.append(surrounding_mc.length)
                features.append(surrounding_mc.get_acceleration())

            ground_truth = DataSet.get_four_classes_groundtruth(mc)
            dataset = DataSet.append_to_dataset(dataset, features, mc, ground_truth, class_labels)

        return dataset

    @staticmethod
    def get_raw_sensor_dataset(measure_collections, dataset=None, is_softmax_y=False):
        class_labels = ['FREE_SPACE', 'PARKING_CAR', 'OVERTAKING_SITUATION', 'PARKING_MC_BC']
        if dataset is None:
            dataset = DataSet('raw_dataset', class_labels, is_softmax_y)

        for i, mc in enumerate(measure_collections):
            features = np.zeros(1024)
            features[:min(1024, len(mc.measures))] = [m.distance for m in mc.measures][:1024]

            np.append(features, mc.avg_speed)
            np.append(features, mc.get_acceleration())

            ground_truth = DataSet.get_four_classes_groundtruth(mc)
            dataset = DataSet.append_to_dataset(dataset, features.tolist(), mc, ground_truth, class_labels)

        return dataset

    @staticmethod
    def get_raw_sensor_dataset_parking_space_detection(measure_collections, dataset=None, is_softmax_y=False):
        class_labels = ['NO_PARKING_CAR', 'PARKING_CAR']
        if dataset is None:
            dataset = DataSet('raw_dataset_parking_space_detection', class_labels, is_softmax_y)

        for i, mc in enumerate(measure_collections):
            features = np.zeros(1024)
            features[:min(1024, len(mc.measures))] = [m.distance for m in mc.measures][:1024]

            np.append(features, mc.avg_speed)
            np.append(features, mc.get_acceleration())

            ground_truth = DataSet.get_parking_classes_groundtruth(mc)
            dataset = DataSet.append_to_dataset(dataset, features.tolist(), mc, ground_truth, class_labels)

        return dataset

    @staticmethod
    def get_raw_sensor_dataset_per_10cm(measure_collections, dataset=None, is_softmax_y=False, name='raw_sensor_dataset_per_10cm'):
        class_labels = ['FREE_SPACE', 'PARKING_CAR', 'OVERTAKING_SITUATION', 'PARKING_MC_BC']
        if dataset is None:
            dataset = DataSet(name, class_labels, is_softmax_y)

        for mc in measure_collections:
            features = np.zeros(100)
            i = 0
            next_length = 0.0
            first_m = mc.measures[0]
            for m in mc.measures:
                if i < len(features):
                    cur_length = m.distance_to(first_m)
                    if next_length <= cur_length:
                        features[i] = m.distance
                        next_length += 0.1
                        i += 1

            np.append(features, mc.avg_speed)
            np.append(features, mc.get_acceleration())

            ground_truth = DataSet.get_four_classes_groundtruth(mc)
            dataset = DataSet.append_to_dataset(dataset, features.tolist(), mc, ground_truth, class_labels)

        return dataset

    # TODO
    @staticmethod
    def get_raw_sensor_dataset_per_10cm_p_surroundings(measure_collections, dataset=None, is_softmax_y=False):
        class_labels = ['FREE_SPACE', 'PARKING_CAR', 'OVERTAKING_SITUATION', 'PARKING_MC_BC']
        if dataset is None:
            dataset = DataSet('raw_dataset_per10cm_p_surrounding', class_labels, is_softmax_y)

        for mc_index, mc in enumerate(measure_collections):
            features = np.zeros(100)
            i = 0
            next_length = 0.0
            first_m = mc.measures[0]
            for m in mc.measures:
                if i < len(features):
                    cur_length = m.distance_to(first_m)
                    if next_length <= cur_length:
                        features[i] = m.distance
                        next_length += 0.1
                        i += 1

            prev_features = DataSet.get_prev_raw_sensor_data_per_10cm(measure_collections, mc_index)

            x = prev_features.tolist()
            x.extend(features)
            x.extend(DataSet.get_next_raw_sensor_data_per_10cm(measure_collections, mc_index).tolist())
            x.append(mc.avg_speed)
            x.append(mc.get_acceleration())

            ground_truth = DataSet.get_four_classes_groundtruth(mc)
            dataset = DataSet.append_to_dataset(dataset, x, mc, ground_truth, class_labels)

        return dataset

    @staticmethod
    def get_prev_raw_sensor_data_per_10cm(measure_collections, mc_index):
        prev_features = np.zeros(100)
        cur_mc = measure_collections[mc_index]
        i = 0
        last_length = -1.0

        if mc_index > 0:
            first_m = measure_collections[mc_index - 1].measures[len(measure_collections[mc_index - 1].measures) - 1]

            prev_index = mc_index - 1
            while prev_index >= 0 and i < len(prev_features):
                mc = measure_collections[prev_index]
                for m in mc.measures[::-1]: # traverse over reverted list of measures
                    if i < len(prev_features):
                        cur_length = m.distance_to(first_m)
                        if last_length + 0.1 <= cur_length:
                            prev_features[len(prev_features) - 1 - i] = m.distance
                            last_length += cur_length
                            i += 1
                prev_index -= 1

        return prev_features

    @staticmethod
    def get_next_raw_sensor_data_per_10cm(measure_collections, mc_index):
        next_features = np.zeros(100)
        cur_mc = measure_collections[mc_index]
        i = 0
        last_length = -1.0
        if mc_index < len(measure_collections) - 1:
            first_m = measure_collections[mc_index + 1].measures[0]

            next_index = mc_index + 1
            while next_index < len(measure_collections) and i < len(next_features):
                mc = measure_collections[next_index]
                for m in mc.measures:
                    if i < len(next_features):
                        cur_length = m.distance_to(first_m)
                        if last_length + 0.1 <= cur_length:
                            next_features[i] = m.distance
                            last_length = cur_length
                            i += 1
                next_index += 1

        return next_features

    def to_arff_file(self, path):
        write_header = not os.path.exists(path)
        is_arff_file = path.endswith('.arff')

        with open(path, 'a') as arff_file:
            if write_header and is_arff_file:
                arff_file.write("@RELATION driveby\n")
                arff_file.write("@ATTRIBUTE avg_distance NUMERIC\n")
                arff_file.write("@ATTRIBUTE length NUMERIC\n")
                arff_file.write("@ATTRIBUTE duration_s NUMERIC\n")
                arff_file.write("@ATTRIBUTE nr_of_measures NUMERIC\n")
                arff_file.write("@ATTRIBUTE distance_variance NUMERIC\n")
                arff_file.write("@ATTRIBUTE avg_speed NUMERIC\n")
                arff_file.write("@ATTRIBUTE avg_accel NUMERIC\n")
                arff_file.write("@ATTRIBUTE diff_prev NUMERIC\n")
                arff_file.write("@ATTRIBUTE diff_next NUMERIC\n")
                arff_file.write("@ATTRIBUTE class {FREE_SPACE, PARKING_CAR, OVERTAKING_SITUATION, PARKING_MC_BC}\n")
                arff_file.write("\n\n\n")
                arff_file.write("@DATA\n")

            for i in range(0, len(self.x)):
                for x_i in self.x[i]:
                    arff_file.write(str(x_i))
                    arff_file.write(",")
                arff_file.write(self.y_true[i])
                arff_file.write("\n")


if __name__ == '__main__':
    base_path = 'C:\\sw\\master\\collected data\\'

    options = {
        'mc_min_speed': 1.0,
        'mc_merge': True,
        'mc_separation_threshold': 1.0,
        'mc_min_measure_count': 2,
        'outlier_threshold_distance': 1.0,
        'outlier_threshold_diff': 0.5,
        'min_measurement_value': 0.06,
    }

    dataset = None
    measure_collections_files_dir = MeasureCollection.read_directory(base_path, options=options)
    measure_collections_dir = {}

    parking_space_map_clusters, _ = create_parking_space_map(measure_collections_files_dir)
    measure_collections_files_dir = filter_parking_space_map_mcs(measure_collections_files_dir,
                                                                 parking_space_map_clusters)

    for file_name, measure_collections in measure_collections_files_dir.items():
        print(file_name)
        dataset = DataSet.get_dataset(measure_collections, dataset=dataset)
        measure_collections_dir.update(MeasureCollection.mc_list_to_dict(measure_collections))

    dataset.to_arff_file('parking_map_filtered_dataset_new.arff')

