
import os
import time

import kivy
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


class GroundTruthTaggingAppStarter(App):

    def __init__(self, **kwargs):
        super(GroundTruthTaggingAppStarter, self).__init__(**kwargs)

        self.fileChooser = TextInput()
        self.submit = Button(text='Tag Ground-Truth')
        self.submit.bind(on_press=self.on_submit)

    def build(self):
        layout = BoxLayout(orientation='vertical')
        # Window.size = (500, 100)
        layout.add_widget(self.fileChooser)
        layout.add_widget(self.submit)

        return layout

    def on_submit(self, instance):
        App.get_running_app().stop()
        GroundTruthTaggingApp(self.fileChooser.text).run()


class GroundTruthTaggingApp(App):

    def __init__(self, base_path, **kwargs):
        super(GroundTruthTaggingApp, self).__init__(**kwargs)

        self.ground_truth = []

        self.base_path = base_path
        self.files = sorted([f for f in os.listdir(self.base_path) if os.path.isfile(os.path.join(self.base_path, f))])
        print(self.files)

        self.cur_index = 0

        self.image = Image(source=os.path.join(self.base_path, self.files[0]), size=(352, 288), pos=(0, 0))
        with self.image.canvas as canvas:
            Color(1., 0, 0)
            Rectangle(pos=(400, 100), size=(1, 400))

        self.button_layout = GridLayout(cols=3, size_hint=(1, .25), pos=(0, 0))
        self.bt_previous = Button(text='<')
        self.bt_previous.bind(on_press=self.on_prev_clicked)
        self.bt_start_here = Button(text='Start from here [s]')
        self.bt_start_here.bind(on_press=self.on_start_here)
        self.bt_par_parking_car = Button(text='Par. Parking Car [p]')
        self.bt_par_parking_car.bind(on_press=self.on_par_parking_car_clicked)
        self.bt_per_parking_car = Button(text='Per. Parking Car [l]')
        self.bt_other_parking_car = Button(text='Other Parking Car [k]')
        self.bt_parking_motorcycle = Button(text='Parking Motorcycle [j]')
        self.bt_parking_bicycle = Button(text='Parking Bicycle [h]')
        self.bt_overtaken_car = Button(text='Overtaken Car [c]')
        self.bt_overtaken_bicycle = Button(text='Overtaken Bicycle [b]')
        self.bt_overtaken_motorcycle = Button(text='Overtaken Motorcycle [m]')
        self.bt_overtaken_car.bind(on_press=self.on_overtaken_car_clicked)
        self.bt_free_space = Button(text='Free Space [f]')
        self.bt_stop_here = Button(text='Stop Here and Save [e]')
        self.bt_stop_here.bind(on_press=self.on_stop_here)
        self.bt_next = Button(text='>')
        self.bt_next.bind(on_press=self.on_next_clicked)
        self.button_layout.add_widget(self.bt_previous)
        self.button_layout.add_widget(self.bt_start_here)
        self.button_layout.add_widget(self.bt_next)

        self._keyboard = Window.request_keyboard(self._keyboard_closed, self)
        self._keyboard.bind(on_key_down=self._on_keyboard_down)

        self.started_index = None

    def build(self):
        layout = FloatLayout(size=(300, 300))
        # Window.size = (1000, 700)
        layout.add_widget(self.image)
        layout.add_widget(self.button_layout)

        return layout

    def _keyboard_closed(self):
        self._keyboard.unbind(on_key_down=self._on_keyboard_down)
        self._keyboard = None

    def _on_keyboard_down(self, keyboard, keycode, text, modifiers):
        if keycode[1] == 'right' and self.started_index is None:
            self.on_next_clicked(None)
        elif keycode[1] == 'left':
            self.on_prev_clicked(None)
        elif keycode[1] == 's' and self.started_index is None:
            self.on_start_here(None)
        elif keycode[1] == 'p' and self.started_index is not None:
            self.on_par_parking_car_clicked(None)
        elif keycode[1] == 'l' and self.started_index is not None:
            self.on_per_parking_car_clicked(None)
        elif keycode[1] == 'k' and self.started_index is not None:
            self.on_other_parking_car_clicked(None)
        elif keycode[1] == 'f' and self.started_index is not None:
            self.on_free_space_clicked(None)
        elif keycode[1] == 'c' and self.started_index is not None:
            self.on_overtaken_car_clicked(None)
        elif keycode[1] == 'b' and self.started_index is not None:
            self.on_overtaken_bicycle_clicked(None)
        elif keycode[1] == 'm' and self.started_index is not None:
            self.on_overtaken_motorcycle_clicked(None)
        elif keycode[1] == 'j' and self.started_index is not None:
            self.on_parking_motorcycle_clicked(None)
        elif keycode[1] == 'h' and self.started_index is not None:
            self.on_parking_bicycle_clicked(None)
        elif keycode[1] == 'e' and self.started_index is not None:
            self.on_stop_here(None)
        return True

    def on_par_parking_car_clicked(self, instance):
        timestamp = self.get_timestamp(self.cur_index)

        gt = GroundTruth(timestamp, GroundTruthClass.PARALLEL_PARKING_CAR)
        self.add_ground_truth(gt)
        self.on_next_clicked('')

    def on_per_parking_car_clicked(self, instance):
        timestamp = self.get_timestamp(self.cur_index)

        gt = GroundTruth(timestamp, GroundTruthClass.PERPENDICULAR_PARKING_CAR)
        self.add_ground_truth(gt)
        self.on_next_clicked('')

    def on_other_parking_car_clicked(self, instance):
        timestamp = self.get_timestamp(self.cur_index)

        gt = GroundTruth(timestamp, GroundTruthClass.OTHER_PARKING_CAR)
        self.add_ground_truth(gt)
        self.on_next_clicked('')

    def on_parking_motorcycle_clicked(self, instance):
        timestamp = self.get_timestamp(self.cur_index)

        gt = GroundTruth(timestamp, GroundTruthClass.PARKING_MOTORCYCLE)
        self.add_ground_truth(gt)
        self.on_next_clicked('')

    def on_parking_bicycle_clicked(self, instance):
        timestamp = self.get_timestamp(self.cur_index)

        gt = GroundTruth(timestamp, GroundTruthClass.PARKING_BICYCLE)
        self.add_ground_truth(gt)
        self.on_next_clicked('')

    def on_overtaken_car_clicked(self, instance):
        timestamp = self.get_timestamp(self.cur_index)

        gt = GroundTruth(timestamp, GroundTruthClass.OVERTAKEN_CAR)
        self.add_ground_truth(gt)
        self.on_next_clicked('')

    def on_overtaken_bicycle_clicked(self, instance):
        timestamp = self.get_timestamp(self.cur_index)

        gt = GroundTruth(timestamp, GroundTruthClass.OVERTAKEN_BICYCLE)
        self.add_ground_truth(gt)
        self.on_next_clicked('')

    def on_overtaken_motorcycle_clicked(self, instance):
        timestamp = self.get_timestamp(self.cur_index)

        gt = GroundTruth(timestamp, GroundTruthClass.OVERTAKEN_MOTORCYCLE)
        self.add_ground_truth(gt)
        self.on_next_clicked('')

    def on_free_space_clicked(self, instance):
        timestamp = self.get_timestamp(self.cur_index)

        gt = GroundTruth(timestamp, GroundTruthClass.FREE_SPACE)
        self.add_ground_truth(gt)
        self.on_next_clicked('')

    def get_timestamp(self, index):
        f = self.files[index]
        dt = datetime.strptime(f.split('.')[0], '%Y%m%d_%H%M%S_%f')
        return (dt - datetime(1970, 1, 1)).total_seconds()

    def add_ground_truth(self, gt):
        index = self.cur_index - self.started_index
        if index >= len(self.ground_truth):
            self.ground_truth.append(gt)
        else:
            self.ground_truth[index] = gt

    def on_next_clicked(self, instance):
        if self.cur_index + 1 < len(self.files):
            self.cur_index += 1
            self.image.source = os.path.join(self.base_path, self.files[self.cur_index])
        else:
            self.on_stop_here(None)

    def on_prev_clicked(self, instance):
        if self.cur_index > 0 and (self.started_index is None or self.cur_index > self.started_index):
            self.cur_index -= 1
            self.image.source = os.path.join(self.base_path, self.files[self.cur_index])

    def on_start_here(self, instance):
        self.button_layout.clear_widgets()
        self.button_layout.add_widget(self.bt_previous)
        self.button_layout.add_widget(self.bt_par_parking_car)
        self.button_layout.add_widget(self.bt_per_parking_car)
        self.button_layout.add_widget(self.bt_other_parking_car)
        self.button_layout.add_widget(self.bt_parking_bicycle)
        self.button_layout.add_widget(self.bt_parking_motorcycle)
        self.button_layout.add_widget(self.bt_free_space)
        self.button_layout.add_widget(self.bt_overtaken_car)
        self.button_layout.add_widget(self.bt_overtaken_bicycle)
        self.button_layout.add_widget(self.bt_overtaken_motorcycle)
        self.button_layout.add_widget(self.bt_stop_here)
        self.started_index = self.cur_index

    def on_stop_here(self, instance):
        GroundTruth.write_to_file(os.path.join(self.base_path, '00gt' + str(time.time()) + '.dat'), self.ground_truth)

        App.get_running_app().stop()




if __name__ == '__main__':
    GroundTruthTaggingAppStarter().run()
