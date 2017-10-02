import csv
from geopy.distance import vincenty

from drive_by_evaluation.ground_truth import GroundTruth


class Measurement:

    def __init__(self, distance, timestamp, latitude, longitude, speed, ground_truth):
        self.distance = distance
        self.timestamp = timestamp
        self.latitude = latitude
        self.longitude = longitude
        self.speed = speed
        self.ground_truth = ground_truth

    def distance_to(self, other_m):
        return vincenty((self.latitude, self.longitude),
                        (other_m.latitude, other_m.longitude)).meters

    @staticmethod
    def read(file_path, ground_truth_path, options=None):
        if options is None:
            options = dict()

        replacement_values = options.get('replacement_values', {})
        min_value = options.get('min_measurement_value', 0.0)

        gps_measurements = []
        distances = []

        with open(file_path, 'r') as captured_file:
            csv_reader = csv.reader(captured_file, delimiter=',')
            i = 0
            last_gps_i = None
            for row in csv_reader:
                sensor_type = row[0]
                timestamp = float(row[1])
                if sensor_type == 'LidarLite':
                    distance_value = float(row[2]) / 100
                    if distance_value >= min_value:
                        distance_value = replacement_values.get(distance_value, distance_value)
                        distances.append(LidarLiteMeasurement(timestamp, distance_value))
                elif sensor_type == 'GPS':
                    if row[2] != '0.0' and row[3] != '0.0' and \
                            row[2] != 'nan' and row[3] != 'nan' and \
                            (last_gps_i is None or (i - last_gps_i) < 3):
                        gps_measurements.append(GPSMeasurement(timestamp, float(row[2]), float(row[3]), float(row[4])))
                    last_gps_i = i
                else:
                    print('unknown sensor', sensor_type)
                i += 1

        print('read gps measures', len(gps_measurements))
        print('read distance measures', len(distances))

        ground_truth = []
        if ground_truth_path is not None:
            ground_truth = GroundTruth.read_from_file(ground_truth_path)

        distance_index = 0
        while gps_measurements[0].timestamp > distances[distance_index].timestamp:
            distance_index += 1
        gps_index = 0

        measurements = []
        while gps_index < len(gps_measurements) - 1:
            g = gps_measurements[gps_index]
            next_g = gps_measurements[gps_index + 1]

            while distance_index < len(distances) and next_g.timestamp > distances[distance_index].timestamp:
                gps = g.get_interpolation(next_g, distances[distance_index].timestamp)
                measurements.append(Measurement(distances[distance_index].distance, distances[distance_index].timestamp,
                                                gps.latitude, gps.longitude, gps.speed, None))
                distance_index += 1

            gps_index += 1

        print('interpolated gps measurements', len(measurements))

        if ground_truth_path is not None:
            measure_index = 0
            ground_truth_index = 0
            while measurements[0].timestamp < ground_truth[0].timestamp:
                measurements.pop(0)

            while ground_truth_index < len(ground_truth):
                gt = ground_truth[ground_truth_index]
                while measure_index < len(measurements) and measurements[measure_index].timestamp < gt.timestamp:
                    measurements[measure_index].ground_truth = gt
                    measure_index += 1

                ground_truth_index += 1

            while measure_index < len(measurements):
                measurements.pop(len(measurements) - 1)

            print('added ground truth', len(measurements))

        print('seconds of measurement', measurements[len(measurements) - 1].timestamp - measurements[0].timestamp)

        outlier_threshold_distance = options.get('outlier_threshold_distance')
        outlier_threshold_diff = options.get('outlier_threshold_diff')
        if outlier_threshold_distance is not None:
            measurements = Measurement.remove_outliers(measurements, outlier_threshold_distance, outlier_threshold_diff)

        return measurements

    @staticmethod
    def remove_outliers(measurements, outlier_threshold_distance, outlier_threshold_diff):
        last_m = measurements[0]
        i = 1
        while i < len(measurements) - 1:
            m = measurements[i]
            next_m = measurements[i+1]

            distance_to_last = m.distance - last_m.distance
            distance_to_next = m.distance - next_m.distance

            if (abs(distance_to_last) > outlier_threshold_distance and
                abs(distance_to_last - distance_to_next) < outlier_threshold_diff):
                measurements.pop(i)
            else:
                i += 1
            last_m = m

        print('filtered outliers', len(measurements))
        return measurements


class LidarLiteMeasurement:
    def __init__(self, timestamp, distance):
        self.distance = distance
        self.timestamp = timestamp


class GPSMeasurement:
    def __init__(self, timestamp, latitude, longitude, speed):
        self.timestamp = timestamp
        self.latitude = latitude
        self.longitude = longitude
        self.speed = speed

    def get_interpolation(self, other_gps, ts):
        lan = self.latitude + (other_gps.latitude - self.latitude)\
                            * (ts - self.timestamp) / (other_gps.timestamp - self.timestamp)
        lon = self.longitude + (other_gps.longitude - self.longitude)\
                             * (ts - self.timestamp) / (other_gps.timestamp - self.timestamp)
        speed = self.speed + (other_gps.speed - self.speed)\
                             * (ts - self.timestamp) / (other_gps.timestamp - self.timestamp)
        return GPSMeasurement(ts, lan, lon, speed)


if __name__ == '__main__':
    measurements = Measurement.read('C:\\sw\\master\\collected data\\data_20170908_linz\\raw_20170908_080745_338288.dat',
                                    'C:\\sw\\master\\collected data\\data_20170908_linz\\raw_20170908_080745_338288.dat_images_Camera\\00gt1505029259.89.dat')



