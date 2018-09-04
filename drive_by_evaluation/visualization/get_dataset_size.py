from drive_by_evaluation.measure_collection import MeasureCollection
import os
from geopy.distance import vincenty

from drive_by_evaluation.measurement import Measurement


def get_raw_dataset_size(base_path, options):
    cnt_measurements = 0
    total_seconds = 0
    total_kms = 0

    for f in sorted(os.listdir(base_path)):
        if os.path.isdir(os.path.join(base_path, f)) and not f.endswith('_images_Camera'):
            sub_dir = os.path.join(base_path, f)
            sub_measurement_cnt, sub_total_seconds, sub_total_kms = get_raw_dataset_size(sub_dir, options)
            cnt_measurements += sub_measurement_cnt
            total_seconds += sub_total_seconds
            total_kms += sub_total_kms
        elif os.path.isfile(os.path.join(base_path, f)) and f.endswith('.dat'):
            data_file = os.path.join(base_path, f)
            camera_folder = data_file + '_images_Camera\\'
            gt_files = [gt_f for gt_f in os.listdir(camera_folder) if gt_f.startswith('00gt')]
            if len(gt_files) > 0:
                measurements = Measurement.read(data_file, os.path.join(camera_folder, gt_files[0]),
                                                options=options)
                cnt_measurements += len(measurements)
                total_seconds += measurements[len(measurements) - 1].timestamp - measurements[0].timestamp
                measure_collections_f = MeasureCollection.create_measure_collections(measurements, options=options)
                for mc in measure_collections_f:
                    length = mc.get_length() / 1000.0
                    if 0.0005 < length < 0.100:
                        total_kms += length
                #total_kms += vincenty((measurements[len(measurements) - 1].latitude, measurements[len(measurements) - 1].longitude)
                #                      (measurements[0].latitude, measurements[0].longitude)).kilometers
                print(total_kms)
                # for i in range(1, len(measurements)):
                #     total_kms += vincenty((measurements[i-1].latitude, measurements[i-1].longitude),
                #                           (measurements[i].latitude, measurements[i].longitude)).meters / 1000.0

    return cnt_measurements, total_seconds, total_kms


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

    measure_collections = {}
    excluded_mcs = options.get('exclude_mcs', [])

    print('')
    raw_size = get_raw_dataset_size(base_path, options)
    print('cnt_measurements, total_seconds')
    print(raw_size)
    print('')

    measure_collections_dir = MeasureCollection.read_directory(base_path, options=options)
    print('')
    print('cnt_measurements, total_seconds, total_kms')
    print(MeasureCollection.get_size(measure_collections_dir))
