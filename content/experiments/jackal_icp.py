import time
import os
import numpy as np

from framework import ros_utils
from framework import utils
from framework import viz_utils
from framework.experiment import Experiment

class JackalIcp(Experiment):

    def clean_env(self):
        utils.kill_process('laser_scan_matcher_node')
        ros_utils.kill_master()

    def task(self, **kwargs):
        ros_utils.start_master(use_sim_time=True)
        ros_utils.launch(package='localization', launch_file='jackal_scan_matcher.launch')
        output_bag_path = os.path.join(self.repetition_dir, '%s_output.bag' % self.name)
        ros_utils.start_recording_bag(output_bag_path, ['/scanmatcher_pose',])
        _, bag_duration = ros_utils.play_bag(self.data_sources, use_clock=True)
        time.sleep(bag_duration)
        ros_utils.stop_recording_bags()
        icp = ros_utils.bag_to_dataframe(output_bag_path, topic=r'/scanmatcher_pose', fields = ['x', 'y'])
        viz_utils.plot_2d_trajectory([icp], labels=['icp'], file_name=os.path.join(self.repetition_dir, 'icp'))
        self.results[self.repetition_id]['icp_error'] = np.linalg.norm(icp.iloc[-1] - icp.iloc[0])