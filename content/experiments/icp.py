import time
import os

import framework.ros_utils as ros_utils
import framework.utils as utils
import framework.viz_utils as viz_utils
from framework.experiment import Experiment

class Icp(Experiment):

    def clean_env(self):
        utils.kill_process('laser_scan_matcher_node')
        ros_utils.kill_master()

    def task(self, **kwargs):
        ros_utils.start_master(use_sim_time=True)
        ros_utils.launch(direct_path=r'/home/omer/orchards_ws/src/air_ground_orchard_navigation/jackal_scan_matcher.launch') # TODO: change
        ros_utils.start_recording_bag(os.path.join(self.repetition_dir, self.name), ['/scanmatcher_pose',])
        _, bag_duration = ros_utils.play_bag(self.data_sources, use_clock=True)

        for _ in range(int(bag_duration)):
            time.sleep(1)


        # time.sleep(bag_duration)
        ros_utils.stop_recording_bags()
        time.sleep(1)


class DrawIcp(Experiment):

    def clean_env(self):
        pass

    def task(self):
        icp = ros_utils.bag_to_dataframe(self.data_sources, topic=r'/scanmatcher_pose', fields = ['x', 'y'])
        viz_utils.plot_2d_trajectory([icp], labels=['icp'], file_name=os.path.join(self.repetition_dir, 'icp'))

if __name__ == '__main__':

    import experiments_framework.content.data_pointers.lavi_april_18.jackal as jackal_data

    bag_descriptor = jackal_data.jackal_18['15-11-01']
    execution_dir = utils.create_new_execution_folder('icp_jackal_18')

    # experiment = Icp(name='icp_test', data_sources=bag_descriptor.path, working_dir=execution_dir)
    # experiment.run(repetitions=1)

    experiment = DrawIcp(name='dummy', data_sources=r'/home/omer/orchards_ws/output/y20180531-173943_icp_jackal_18/20180531-173943_icp_test/0icp_test.bag', working_dir=execution_dir)
    experiment.run(repetitions=1)

    # TODO: check different combination: with/without imu and odometry
    # TODO: check combinations in EKF (add icp as additional input, choose a few typical configurations)
    # TODO: check for errors (final + waypoint)
    # TODO: gradual plotting (just like in EKF)
    # TODO: put this and EKF on the same plot