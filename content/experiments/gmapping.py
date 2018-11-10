import time
import datetime

from framework.experiment import Experiment
from framework import ros_utils
from framework import utils

class Gmapping(Experiment):

    def clean_env(self):
        utils.kill_process('slam_gmapping')
        ros_utils.kill_master()


    def task(self, **kwargs):
        ros_utils.start_master(use_sim_time=True)
        ros_utils.launch(package='jackal_navigation', launch_file='gmapping.launch')
        _, bag_duration = ros_utils.play_bag(self.data_sources, use_clock=True)
        start_time = datetime.datetime.now()
        while True:
            if (datetime.datetime.now() - start_time).seconds > (bag_duration + 1):
                break
            if kwargs.get('periodic_map_saving'):
                ros_utils.save_map(map_name=datetime.datetime.now().strftime('map_%H_%M_%S'), dir_name=self.repetition_dir)
            time.sleep(3)
        ros_utils.save_map(map_name=datetime.datetime.now().strftime('map_%H_%M_%S-final'), dir_name=self.repetition_dir) # TODO: verify that this works


