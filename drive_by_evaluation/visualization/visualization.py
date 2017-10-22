
import os
import kivy
from kivy.clock import Clock
from kivy.garden.graph import Graph, MeshLinePlot
from kivy.graphics import Color
from kivy.graphics import Rectangle
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.gridlayout import GridLayout
from kivy.uix.floatlayout import FloatLayout
from kivy.uix.image import Image
from kivy.uix.textinput import TextInput
from sklearn.ensemble import RandomForestClassifier

from drive_by_evaluation.db_machine_learning.db_data_set import DataSet

kivy.require('1.10.0')

from kivy.app import App
from kivy.uix.button import Button
from kivy.core.window import Window

import numpy as np

from datetime import datetime

from drive_by_evaluation.ground_truth import GroundTruth, GroundTruthClass
from drive_by_evaluation.measure_collection import MeasureCollection
from drive_by_evaluation.measurement import Measurement


class VisualizationAppStarter(App):

    def __init__(self, **kwargs):
        super(VisualizationAppStarter, self).__init__(**kwargs)

        self.fileChooser = TextInput()
        self.submit = Button(text='Start Visualization')
        self.submit.bind(on_press=self.on_submit)

    def build(self):
        layout = BoxLayout(orientation='vertical')
        # Window.size = (500, 100)
        layout.add_widget(self.fileChooser)
        layout.add_widget(self.submit)

        return layout

    def on_submit(self, instance):
        App.get_running_app().stop()
        VisualizationApp(self.fileChooser.text).run()


class VisualizationApp(App):

    def __init__(self, data_file, measure_collections_f, additional_interval, **kwargs):
        super(VisualizationApp, self).__init__(**kwargs)

        self.data_file = data_file
        self.additional_interval = additional_interval
        self.camera_folder = data_file + '_images_Camera\\'
        self.camera_files = sorted([f for f in os.listdir(self.camera_folder)
                                    if os.path.isfile(os.path.join(self.camera_folder, f)) and f.endswith('.jpg')])
        self.ground_truth_file = [os.path.join(self.camera_folder, f) for f in os.listdir(self.camera_folder)
                                  if os.path.isfile(os.path.join(self.camera_folder, f)) and f.startswith('00gt')][0]

        self.image = Image(source=os.path.join(self.camera_folder, self.camera_files[0]), size=(352, 288), pos=(0, 0))
        #with self.image.canvas as canvas:
        #    Color(1., 0, 0)
         #   Rectangle(size=(1, 10000))

        self.graph = Graph(xlabel='Time [s]', ylabel='Distance [m]', #x_ticks_minor=0.5,
                           x_ticks_major=2, y_ticks_major=1,
                           y_grid_label=True, x_grid_label=True, padding=10,
                           x_grid=True, y_grid=True, xmin=0, xmax=0, ymin=-1, ymax=11)

        last_mc = None
        self.first_timestamp = measure_collections_f[0].first_measure().timestamp
        for mc in measure_collections_f:
            color = [1, 1, 0, 1]
            if mc.prediction == 'FREE_SPACE':
                color = [1, 0, 1, 1]
            elif mc.prediction == 'PARKING_CAR':
                color = [0, 1, 1, 1]
            elif mc.prediction == 'OVERTAKING_SITUATION':
                color = [0, 0, 1, 1]
            plot = MeshLinePlot(color=color)
            plot.points = [(m.timestamp - self.first_timestamp, m.distance) for m in mc.measures]
            self.graph.add_plot(plot)

            color_actual = [1, 1, 0, 1]
            if mc.get_probable_ground_truth() == GroundTruthClass.FREE_SPACE:
                color_actual = [1, 0, 1, 1]
            elif GroundTruthClass.is_parking_car(mc.get_probable_ground_truth()):
                color_actual = [0, 1, 1, 1]
            elif GroundTruthClass.is_overtaking_situation(mc.get_probable_ground_truth()):
                color_actual = [0, 0, 1, 1]
            plot_actual = MeshLinePlot(color=color_actual)
            plot_actual.points = [(m.timestamp - self.first_timestamp, m.distance - 0.1) for m in mc.measures]
            self.graph.add_plot(plot_actual)

            if last_mc is not None:
                plot_next = MeshLinePlot(color=[1, 1, 1, 1])
                plot_next.points = [(last_mc.last_measure().timestamp - self.first_timestamp, last_mc.last_measure().distance),
                                    (mc.first_measure().timestamp - self.first_timestamp, mc.first_measure().distance)]
                self.graph.add_plot(plot_next)

            last_mc = mc

        # plot = MeshLinePlot(color=[1, 1, 1, 1])
        # plot.points = [(m.timestamp - self.first_timestamp, m.distance) for m in self.measurements]
        # self.graph.add_plot(plot)

        self._keyboard = Window.request_keyboard(self._keyboard_closed, self)
        self._keyboard.bind(on_key_down=self._on_keyboard_down)

        self.running = True
        self.cur_index = -1
        self.show_next_image(0)

    def build(self):
        flow_layout = FloatLayout()
        layout = BoxLayout(size=(300, 300), orientation='vertical')
        # Window.size = (1000, 700)
        layout.add_widget(self.image)
        layout.add_widget(self.graph)

        flow_layout.add_widget(layout)
        with flow_layout.canvas as canvas:
            Color(1., 0, 0)
            Rectangle(size=(1, 10000))
        return flow_layout

    def _keyboard_closed(self):
        self._keyboard.unbind(on_key_down=self._on_keyboard_down)
        self._keyboard = None

    def _on_keyboard_down(self, keyboard, keycode, text, modifiers):
        if keycode[1] == 'spacebar':
            self.on_start_stop()
        return True

    def on_start_stop(self):
        if not self.running:
            self.running = True
            self.show_next_image(0)
        else:
            self.running = False

    def show_next_image(self, dt):
        self.cur_index += 1
        self.image.source = os.path.join(self.camera_folder, self.camera_files[self.cur_index])
        cur_time = self.get_timestamp(self.cur_index)

        self.graph.xmin = cur_time - self.first_timestamp - 2
        self.graph.xmax = cur_time - self.first_timestamp + 2

        #self.graph.export_to_png('C:\\sw\\master\\scenario_snapshots\\export_' + self.camera_files[self.cur_index] + '.png')

        next_dt = 1
        if self.cur_index + 1 < len(self.camera_files):
            next_time = self.get_timestamp(self.cur_index + 1)
            next_dt = next_time - cur_time + self.additional_interval
        else:
            self.cur_index = -1

        if self.running:
            Clock.schedule_once(self.show_next_image, next_dt)

    def get_timestamp(self, index):
        f = self.camera_files[index]
        dt = datetime.strptime(f.split('.')[0], '%Y%m%d_%H%M%S_%f')
        return (dt - datetime(1970, 1, 1)).total_seconds()


if __name__ == '__main__':
    #scenario_path = 'C:\\sw\\master\\scenarios\\overtaking_cars_and_perpendicular_cars.dat'
    #additional_timeout = 0.02
    scenario_path = 'C:\\sw\\master\\scenarios\\parking_cars.dat'
    additional_timeout = 0.01
    #scenario_path = 'C:\\sw\\master\\scenarios\\parking_cars_angular.dat'
    #additional_timeout = 0.01
    #scenario_path = 'C:\\sw\\master\\scenarios\\overtaking_bike.dat'
    #additional_timeout = 0.01

    base_path = 'C:\\sw\\master\\collected data\\'

    options = {
        'mc_min_speed': 3.0,
        'mc_merge': True,
        'mc_separation_threshold': 1.0,
        'mc_min_measure_count': 2,
        # 'mc_surrounding_times_s': [2.0, 5.0],
        'outlier_threshold_distance': 1.0,
        'outlier_threshold_diff': 0.5,
        'max_measure_value': 10.0,
        # 'replacement_values': {0.01: 10.01},
        'min_measurement_value': 0.06
    }

    camera_folder = scenario_path + '_images_Camera\\'
    ground_truth_file = [os.path.join(camera_folder, f) for f in os.listdir(camera_folder)
                              if os.path.isfile(os.path.join(camera_folder, f)) and f.startswith('00gt')][0]
    measurements_scenario = Measurement.read(scenario_path, ground_truth_file, options=options)
    measure_collections_scenario = MeasureCollection.create_measure_collections(measurements_scenario, options=options)
    dataset_scenario = DataSet.get_dataset(measure_collections_scenario)

    options['exclude_mcs'] = measure_collections_scenario

    dataset = None
    measure_collections_files_dir = MeasureCollection.read_directory(base_path, options=options)
    for file_name, measure_collections in measure_collections_files_dir.items():
        dataset = DataSet.get_dataset(measure_collections, dataset=dataset)

    clf = RandomForestClassifier(n_estimators=1000, n_jobs=-1, random_state=42)
    clf.fit(dataset.x, dataset.y_true)

    predictions = clf.predict(np.array(dataset_scenario.x)).reshape(1, -1)[0]

    #print(predictions[0])
    #print(dataset_scenario.y_true[0])
    nr_false = 0
    for i in range(0, len(dataset_scenario.y_true)):
        measure_collections_scenario[i].prediction = predictions[i]
        if predictions[i] != dataset_scenario.y_true[i]:
            print('Predicted: ', predictions[i])
            print('GroundTruth: ', dataset_scenario.y_true[i])
            print('features: ', dataset_scenario.x[i])
            print('')
            nr_false += 1

    print(nr_false)
    print(len(dataset_scenario.y_true))

    VisualizationApp(scenario_path, measure_collections_scenario, additional_interval=additional_timeout).run()




