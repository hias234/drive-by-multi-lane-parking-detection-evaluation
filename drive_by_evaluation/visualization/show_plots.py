
from scipy.signal import butter, lfilter
import matplotlib.pyplot as plt
import matplotlib
import gmplot

from drive_by_evaluation.ground_truth import GroundTruthClass
from drive_by_evaluation.measure_collection import MeasureCollection


class MeasurementVisualization:

    def show_distance_signal(self, measurements, fig=None, base_timestamp=None, color='b'):
        if fig is None:
            fig = plt.figure(4)
        if base_timestamp is None:
            base_timestamp = 0
        # xs = []
        # length = 0
        # for index in range(1, len(measurements)):
        #     xs.append(length)
        #     length += vincenty(
        #                     (measurements[index - 1].latitude, measurements[index - 1].longitude),
        #                     (measurements[index].latitude, measurements[index].longitude)
        #                 ).meters
        # xs.append(length)
        xs = [raw.timestamp - base_timestamp for raw in measurements]
        ys = [raw.distance for raw in measurements]

        plt.plot(xs, ys, c=color)
        plt.ylabel('Distance [m]')
        plt.xlabel('Time [s]')

        #median_distance = np.mean(ys)
        #plt.plot([xs[0], xs[len(xs) - 1]], [median_distance, median_distance])

        fig.show()

    def show_distance_signal_low_pass(self, measurements,  fig=None):
        if fig is None:
            fig = plt.figure(5)

        xs = [raw.timestamp for raw in measurements]
        ys = [raw.distance for raw in measurements]

        fs = 10E9  # 1 ns -> 1 GHz
        cutoff = 10E6  # 10 MHz
        B, A = butter(1, cutoff / (fs / 2), btype='low')  # 1st order Butterworth low-pass
        ys_filtered = lfilter(B, A, ys, axis=0)

        #bz, az = scipy.signal.butter(0, 1 / (200 / 2))  # Gives you lowpass Butterworth as default
        #ys_filtered = scipy.signal.filtfilt(bz, az, input)  # Makes forward/reverse filtering (linear phase filter)

        plt.plot(xs, ys)
        plt.plot(xs, ys_filtered)

        fig.show()

    def show_distance_signal_scatter(self, measurements, fig=None, base_timestamp=None):
        if fig is None:
            fig = plt.figure(1)
        if base_timestamp is None:
            base_timestamp = 0

        xs = [raw.timestamp - base_timestamp for raw in measurements]
        ys = [raw.distance for raw in measurements]
        #speeds = [raw.speed * 100 * 3.6 for raw in measurements]
        cs = self.get_color_list(measurements)

        plt.scatter(xs, ys, c=cs)
        #plt.plot(xs, speeds, c='blue')
        plt.ylabel('distance [m]', fontsize=16)
        plt.xlabel('time [s]', fontsize=16)

        fig.show()

    def get_color_list(self, measurements):
        cs = []
        for raw in measurements:
            if GroundTruthClass.is_parking_car(raw.ground_truth.ground_truth_class):
                cs.append('g')
            elif GroundTruthClass.is_overtaking_situation(raw.ground_truth.ground_truth_class):
                cs.append('r')
            elif GroundTruthClass.is_parking_motorcycle_or_bicycle(raw.ground_truth.ground_truth_class):
                cs.append('c')
            else:
                cs.append('y')
        return cs

    def show_3d(self, measurements, fig=None):
        if fig is None:
            fig = plt.figure(2)
        ax = fig.add_subplot(111, projection='3d')

        xs = [raw.latitude for raw in measurements]
        ys = [raw.longitude for raw in measurements]
        zs = [raw.distance for raw in measurements]
        cs = self.get_color_list(measurements)

        ax.scatter(xs, ys, zs, 'z', c=cs, depthshade=True)

        ax.set_xlabel('Latitude')
        ax.set_ylabel('Longitude')
        ax.set_zlabel('Distance')

        fig.show()

    def show_gps_locations(self, measurements):
        gmap = gmplot.GoogleMapPlotter(48.3045, 14.291153333, 16)
        gmap.scatter([raw.latitude for raw in measurements], [raw.longitude for raw in measurements],
                     '#3B0B39', size=1, marker=False)

        gmap.draw("C:\\sw\\master\\mymap1.html")

    def show_gps_locations_mc(self, mc_dir):
        gmap = gmplot.GoogleMapPlotter(48.3045, 14.291153333, 16)
        for name, mc_list in mc_dir.items():
            gmap.scatter([mc.center_latitude for mc in mc_list if GroundTruthClass.is_parking_car(mc.get_probable_ground_truth())],
                         [mc.center_longitude for mc in mc_list if GroundTruthClass.is_parking_car(mc.get_probable_ground_truth())],
                         '#00FF00', size=2, marker=False)
            gmap.scatter([mc.center_latitude for mc in mc_list if not GroundTruthClass.is_parking_car(mc.get_probable_ground_truth())],
                         [mc.center_longitude for mc in mc_list if not GroundTruthClass.is_parking_car(mc.get_probable_ground_truth())],
                         '#000000', size=1, marker=False)

        gmap.draw("C:\\sw\\master\\mymap_mcs.html")

    def show_distances_plus_segmentation(self, measure_collections, fig=None):
        if fig is None:
            fig = plt.figure(8)
        for measure_collection in measure_collections:
            self.show_distance_signal_scatter(measure_collection.measures, fig=fig)
            xs = [measure_collection.first_measure().timestamp, measure_collection.last_measure().timestamp]
            ys = [measure_collection.first_measure().distance, measure_collection.last_measure().distance]
            probable_gt = measure_collection.get_probable_ground_truth()
            color = 'black'
            if GroundTruthClass.is_parking_car(probable_gt):
                color = 'orange'
            elif GroundTruthClass.is_overtaking_situation(probable_gt):
                color = 'magenta'
            elif GroundTruthClass.is_parking_motorcycle_or_bicycle(probable_gt):
                color = 'yellow'
            plt.plot(xs, ys, color=color)
            plt.scatter(xs, ys, color='black', s=5)

            #plt.scatter([measure_collection.first_measure().timestamp], [measure_collection.get_acceleration() * 1000], color='orange')
        fig.show()

    def show_2d_scatter(self, measure_collections, fig=None):
        if fig is None:
            fig = plt.figure(500)
        for measure_collection in measure_collections:
            xs = [measure_collection.length]
            ys = [measure_collection.avg_distance]
            probable_gt = measure_collection.get_probable_ground_truth()
            color = 'black'
            if GroundTruthClass.is_parking_car(probable_gt):
                color = 'orange'
            elif GroundTruthClass.is_overtaking_situation(probable_gt):
                color = 'magenta'
            elif GroundTruthClass.is_parking_motorcycle_or_bicycle(probable_gt):
                color = 'yellow'
            plt.scatter(xs, ys, color=color, s=5)

            # plt.scatter([measure_collection.first_measure().timestamp], [measure_collection.get_acceleration() * 1000], color='orange')
        fig.show()

    def show_distance_for_class(self, measure_collections, classes, fig=None):
        if fig is None:
            fig = plt.figure(9)
        i = 0
        for measure_collection in measure_collections:
            if measure_collection.get_probable_ground_truth() in classes:
                j = 0 if i < 10 else i - 10
                base_timestamp = measure_collections[j].first_measure().timestamp
                stop = len(measure_collections) if i + 10 >= len(measure_collections) else i + 10
                measures = []
                while j <= stop:
                    measures.extend(measure_collections[j].measures)
                    j += 1
                self.show_distance_signal(measures, fig=fig, base_timestamp=base_timestamp, color='black')
                #self.show_distance_signal(measure_collections[i].measures, fig=fig, color='g',
                #                                  base_timestamp=base_timestamp)
                j = 0 if i < 10 else i - 10
                while j <= stop:
                    if measure_collections[j].get_probable_ground_truth() in classes:
                        self.show_distance_signal(measure_collections[j].measures, fig=fig, color='r', base_timestamp=base_timestamp)
                    j += 1
                fig = plt.figure(i*100 + i*10 + i)
            i += 1

    def show_distance_histogram_length(self, measure_collections, fig=None):
        if fig is None:
            fig = plt.figure(10)
        x = [measure_collection.length for measure_collection in measure_collections]
        n, bins, patches = plt.hist(x, bins=100)
        plt.xlabel('Length [m]')
        plt.ylabel('Count')


if __name__ == '__main__':
    visualization = MeasurementVisualization()
    # base_path = 'C:\\sw\\master\\collected data\\data_20170725_linz\\'
    #base_path = 'C:\\sw\\master\\collected data\\data_20170908_linz\\'
    base_path = 'C:\\sw\\master\\collected data\\'

    font = {'family': 'normal',
            'weight': 'normal',
            'size': 30}

    matplotlib.rc('font', **font)

    options = {
        'mc_min_speed': 3.0,
        'mc_merge': True,
        'mc_separation_threshold': 1.0,
        'mc_min_measure_count': 2,
        #'mc_surrounding_times_s': [2.0, 5.0],
        'outlier_threshold_distance': 1.0,
        'outlier_threshold_diff': 0.5,
        'replacement_values': {0.01: 10.01},
        'min_measurement_value': 0.06,
    }

    # measurements = Measurement.read('C:\\sw\\master\\collected data\\data_20170718_tunnel\\raw_20170718_074348_696382.dat',
    #                                 'C:\\sw\\master\\collected data\\data_20170718_tunnel\\raw_20170718_074348_696382.dat_images_Camera\\00gt1500721683.81.dat',
    #                                 options)
    #
    # visualization.show_distance_signal(measurements, plt.figure(1))
    # visualization.show_distance_signal_scatter(measurements, plt.figure(2))

    #free_space_measure_collections = []
    measure_collections_dir = MeasureCollection.read_directory(base_path, options=options)
    #gt25 = 0
    i = 1
    for file_name, measure_collection in measure_collections_dir.items():
        #visualization.show_2d_scatter(measure_collection, fig=plt.figure(1))
        #visualization.show_distances_plus_segmentation(measure_collection, fig=plt.figure(i))
        #visualization.show_distance_for_class(measure_collection, [GroundTruthClass.OVERTAKEN_BICYCLE], fig=plt.figure(i))
        #gt25 += len([mc for mc in measure_collection if mc.get_probable_ground_truth() == GroundTruthClass.FREE_SPACE and mc.length >= 5])
        #free_space_measure_collections.extend([mc for mc in measure_collection if mc.get_probable_ground_truth() == GroundTruthClass.FREE_SPACE]);
        i += 1
    #
    # print gt25
    # visualization.show_distance_histogram_length(free_space_measure_collections, fig=plt.figure(i))
    #measure_collections = MeasureCollection.read_from_file('C:\\sw\\master\\collected data\\data_20170707\\tagged_mc_20170705_065613_869794.dat')
    #visualization.show_distances_plus_segmentation(measure_collections)
    #visualization.show_distance_signal(measurements)
    #visualization.show_3d(measurements)
    #visualization.show_gps_locations(measurements)
    plt.show()

    visualization.show_gps_locations_mc(measure_collections_dir)
