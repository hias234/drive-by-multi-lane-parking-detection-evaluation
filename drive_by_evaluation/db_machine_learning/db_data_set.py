from drive_by_evaluation.ground_truth import GroundTruthClass
import numpy as np


class DataSet:
    def __init__(self, class_labels):
        self.x = []
        self.y_true = []
        self.class_labels = class_labels

    def append_sample(self, x, y_true):
        self.x.append(x)
        self.y_true.append(y_true)

    def get_class_weights(self, use_floats=False):
        class_weights = {}
        for y in self.y_true:
            if use_floats:
                for y_i in y:
                    class_weights[y_i] = class_weights.get(y_i, 0) + y_i
            else:
                class_weights[y] = class_weights.get(y, 0) + 1
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
    def append_to_dataset(dataset, features, ground_truth, class_labels, use_floats):
        if use_floats:
            dataset.append_sample(features, class_labels.index(ground_truth))
        else:
            dataset.append_sample(features, ground_truth)

        return dataset

    @staticmethod
    def get_dataset(measure_collections, dataset=None, use_floats=False):
        class_labels = ['FREE_SPACE', 'PARKING_CAR', 'OVERTAKING_SITUATION', 'PARKING_MC_BC']
        if dataset is None:
            dataset = DataSet(class_labels)

        for i, mc in enumerate(measure_collections):
            last_distance = 0 if i == 0 else measure_collections[i - 1].avg_distance  # .last_measure().distance
            next_distance = 0 if len(measure_collections) == i + 1 else measure_collections[
                i + 1].avg_distance  # .first_measure().distance
            features = [mc.id,
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
                        mc.first_measure().distance - last_distance,
                        mc.last_measure().distance - next_distance,
                        mc.first_measure().distance,
                        mc.measures[int(len(mc.measures) / 2)].distance,
                        mc.measures[int(len(mc.measures) / 4)].distance,
                        mc.measures[int(len(mc.measures) / 4 * 3)].distance,
                        mc.last_measure().distance
                        ]

            for interval, surrounding_mc in mc.time_surrounding_mcs.items():
                features.append(surrounding_mc.avg_distance)
                features.append(surrounding_mc.avg_speed)
                features.append(surrounding_mc.length)
                features.append(surrounding_mc.get_acceleration())

            ground_truth = DataSet.get_four_classes_groundtruth(mc)
            dataset = DataSet.append_to_dataset(dataset, features, ground_truth, class_labels, use_floats)

        return dataset

    @staticmethod
    def get_raw_sensor_dataset(measure_collections, dataset=None, use_floats=False):
        class_labels = ['FREE_SPACE', 'PARKING_CAR', 'OVERTAKING_SITUATION', 'PARKING_MC_BC']
        if dataset is None:
            dataset = DataSet(class_labels)

        for i, mc in enumerate(measure_collections):
            features = np.zeros(1024)
            features[:min(1024, len(mc.measures))] = [m.distance for m in mc.measures][:1024]

            np.append(features, mc.avg_speed)
            np.append(features, mc.get_acceleration())

            ground_truth = DataSet.get_four_classes_groundtruth(mc)
            dataset = DataSet.append_to_dataset(dataset, features.tolist(), ground_truth, class_labels, use_floats)

        return dataset

    @staticmethod
    def get_raw_sensor_dataset_parking_space_detection(measure_collections, dataset=None, use_floats=False):
        class_labels = ['NO_PARKING_CAR', 'PARKING_CAR']
        if dataset is None:
            dataset = DataSet(class_labels)

        for i, mc in enumerate(measure_collections):
            features = np.zeros(1024)
            features[:min(1024, len(mc.measures))] = [m.distance for m in mc.measures][:1024]

            np.append(features, mc.avg_speed)
            np.append(features, mc.get_acceleration())

            ground_truth = DataSet.get_parking_classes_groundtruth(mc)
            dataset = DataSet.append_to_dataset(dataset, features.tolist(), ground_truth, class_labels, use_floats)

        return dataset

    @staticmethod
    def get_raw_sensor_dataset_per_10cm(measure_collections, dataset=None, use_floats=False):
        class_labels = ['FREE_SPACE', 'PARKING_CAR', 'OVERTAKING_SITUATION', 'PARKING_MC_BC']
        if dataset is None:
            dataset = DataSet(class_labels)

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
            dataset = DataSet.append_to_dataset(dataset, features.tolist(), ground_truth, class_labels, use_floats)

        return dataset
