from enum import Enum
import csv


class GroundTruthClass(Enum):
    FREE_SPACE = 1
    PARALLEL_PARKING_CAR = 2
    PERPENDICULAR_PARKING_CAR = 3
    OTHER_PARKING_CAR = 4
    OVERTAKEN_CAR = 5
    OVERTAKEN_BICYCLE = 6
    OVERTAKEN_MOTORCYCLE = 7
    PARKING_MOTORCYCLE = 8
    PARKING_BICYCLE = 9

    @staticmethod
    def is_parking_car(ground_truth_class):
        return ground_truth_class == GroundTruthClass.PARALLEL_PARKING_CAR or \
               ground_truth_class == GroundTruthClass.PERPENDICULAR_PARKING_CAR or \
               ground_truth_class == GroundTruthClass.OTHER_PARKING_CAR

    @staticmethod
    def is_overtaking_situation(ground_truth_class):
        return ground_truth_class == GroundTruthClass.OVERTAKEN_CAR or \
               ground_truth_class == GroundTruthClass.OVERTAKEN_BICYCLE or \
               ground_truth_class == GroundTruthClass.OVERTAKEN_MOTORCYCLE

    @staticmethod
    def is_parking_motorcycle_or_bicycle(ground_truth_class):
        return ground_truth_class == GroundTruthClass.PARKING_MOTORCYCLE or \
               ground_truth_class == GroundTruthClass.PARKING_BICYCLE


class GroundTruth:

    def __init__(self, timestamp, ground_truth_class):
        self.timestamp = timestamp
        self.ground_truth_class = ground_truth_class

    def is_parking_car(self):
        return GroundTruthClass.is_parking_car(self.ground_truth_class)

    def is_overtaking_situation(self):
        return GroundTruthClass.is_overtaking_situation(self.ground_truth_class)

    def is_parking_motorcycle_or_bicycle(self):
        return GroundTruthClass.is_parking_motorcycle_or_bicycle(self.ground_truth_class)

    @staticmethod
    def read_from_file(ground_truth_path):
        ground_truth = []
        with open(ground_truth_path, 'r') as gt_file:
            csv_reader = csv.reader(gt_file, delimiter=',')

            for row in csv_reader:
                if len(row) > 0:
                    timestamp = float(row[0])
                    ground_truth_class = GroundTruthClass[row[1]]
                    ground_truth.append(GroundTruth(timestamp, ground_truth_class))

        return ground_truth

    @staticmethod
    def write_to_file(ground_truth_path, ground_truth):
        with open(ground_truth_path, 'a') as out:
            csv_out = csv.writer(out)

            for gt in ground_truth:
                print(gt.timestamp, gt.ground_truth_class.name)
                csv_out.writerow([gt.timestamp, gt.ground_truth_class.name])
