from drive_by_evaluation.measure_collection import MeasureCollection
import os

from drive_by_evaluation.measurement import Measurement


def get_raw_dataset_size(base_path, options):
    cnt_measurements = 0
    total_seconds = 0

    for f in sorted(os.listdir(base_path)):
        if os.path.isdir(os.path.join(base_path, f)) and not f.endswith('_images_Camera'):
            sub_dir = os.path.join(base_path, f)
            sub_measurement_cnt, sub_total_seconds = get_raw_dataset_size(sub_dir, options)
            cnt_measurements += sub_measurement_cnt
            total_seconds += sub_total_seconds
        elif os.path.isfile(os.path.join(base_path, f)) and f.endswith('.dat'):
            data_file = os.path.join(base_path, f)
            camera_folder = data_file + '_images_Camera\\'
            gt_files = [gt_f for gt_f in os.listdir(camera_folder) if gt_f.startswith('00gt')]
            if len(gt_files) > 0:
                measurements = Measurement.read(data_file, os.path.join(camera_folder, gt_files[0]),
                                                options=options)
                cnt_measurements += len(measurements)
                total_seconds += measurements[len(measurements) - 1].timestamp - measurements[0].timestamp

    return cnt_measurements, total_seconds


if __name__ == '__main__':
    base_path = 'C:\\sw\\master\\collected data\\'

    options = {
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
    print('cnt_measure_collections, cnt_measurements, total_seconds')
    print(MeasureCollection.get_size(measure_collections_dir))
