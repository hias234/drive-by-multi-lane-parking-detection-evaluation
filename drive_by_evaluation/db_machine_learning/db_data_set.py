from drive_by_evaluation.ground_truth import GroundTruthClass


class DataSet:
    def __init__(self, class_labels):
        self.x = []
        self.y_true = []
        self.class_labels = class_labels

    def append_sample(self, x, y_true):
        self.x.append(x)
        self.y_true.append(y_true)

    @staticmethod
    def get_dataset(measure_collections, dataset=None, use_floats=False):
        if dataset is None:
            dataset = DataSet(['FREE_SPACE', 'PARKING_CAR', 'OVERTAKING_SITUATION', 'PARKING_MC_BC'])

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

            ground_truth = 'FREE_SPACE'
            gt = mc.get_probable_ground_truth()
            if GroundTruthClass.is_parking_car(gt):
                ground_truth = 'PARKING_CAR'
            elif GroundTruthClass.is_overtaking_situation(gt):
                ground_truth = 'OVERTAKING_SITUATION'
            elif GroundTruthClass.is_parking_motorcycle_or_bicycle(gt):
                ground_truth = 'PARKING_MC_BC'

            if use_floats:
                dataset.append_sample(features, ['FREE_SPACE', 'PARKING_CAR', 'OVERTAKING_SITUATION', 'PARKING_MC_BC']
                                                 .index(ground_truth))
            else:
                dataset.append_sample(features, ground_truth)

        return dataset