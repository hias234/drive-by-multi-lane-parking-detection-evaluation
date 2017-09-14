
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

kivy.require('1.10.0')

from kivy.app import App
from kivy.uix.button import Button
from kivy.core.window import Window

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

    def __init__(self, data_file, **kwargs):
        super(VisualizationApp, self).__init__(**kwargs)

        self.data_file = data_file
        self.camera_folder = data_file + '_images_Camera\\'
        self.camera_files = sorted([f for f in os.listdir(self.camera_folder)
                                    if os.path.isfile(os.path.join(self.camera_folder, f)) and f.endswith('.jpg')])
        self.ground_truth_file = [os.path.join(self.camera_folder, f) for f in os.listdir(self.camera_folder)
                                  if os.path.isfile(os.path.join(self.camera_folder, f)) and f.startswith('00gt')][0]
        #print self.camera_files

        options = {
            'mc_min_speed': 4.0, 'mc_merge': True,
            'mc_separation_threshold': 1.0, 'mc_min_measure_count': 2,
            # 'mc_surrounding_times_s': [10.0],
            # 'mc_surrounding_m': [50.0, 100.0],
            'outlier_threshold_distance': 1.0, 'outlier_threshold_diff': 0.5,
            '1cm_replacement_value': 10.01
        }

        self.measurements = Measurement.read(data_file, self.ground_truth_file, options=options)
        self.measure_collections_f = MeasureCollection.create_measure_collections(self.measurements, options=options)

        self.image = Image(source=os.path.join(self.camera_folder, self.camera_files[0]), size=(352, 288), pos=(0, 0))
        #with self.image.canvas as canvas:
        #    Color(1., 0, 0)
         #   Rectangle(size=(1, 10000))

        self.graph = Graph(xlabel='Time [s]', ylabel='Distance [m]', #x_ticks_minor=0.5,
                           x_ticks_major=2, y_ticks_major=1,
                           y_grid_label=True, x_grid_label=True, padding=10,
                           x_grid=True, y_grid=True, xmin=0, xmax=0, ymin=-1, ymax=11)
        self.first_timestamp = self.measurements[0].timestamp
        plot = MeshLinePlot(color=[1, 1, 1, 1])
        plot.points = [(m.timestamp - self.first_timestamp, m.distance) for m in self.measurements]
        self.graph.add_plot(plot)

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

    def show_next_image(self, dt):
        self.cur_index += 1
        self.image.source = os.path.join(self.camera_folder, self.camera_files[self.cur_index])
        cur_time = self.get_timestamp(self.cur_index)

        self.graph.xmin = cur_time - self.first_timestamp - 2
        self.graph.xmax = cur_time - self.first_timestamp + 2

        if self.cur_index + 1 < len(self.camera_files):
            next_time = self.get_timestamp(self.cur_index + 1)
            Clock.schedule_once(self.show_next_image, next_time - cur_time)

    def get_timestamp(self, index):
        f = self.camera_files[index]
        dt = datetime.strptime(f.split('.')[0], '%Y%m%d_%H%M%S_%f')
        return (dt - datetime(1970, 1, 1)).total_seconds()


if __name__ == '__main__':
    VisualizationApp('C:\\sw\\master\\collected data\\data_20170725_linz\\raw_20170725_065205_690205.dat').run()

