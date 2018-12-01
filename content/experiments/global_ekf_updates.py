import time
import os

from framework.experiment import Experiment
from framework import ros_utils
from framework import viz_utils

class GlobalEkfUpdatesExperiment(Experiment):

    def clean_env(self):
        ros_utils.kill_master()

    def task(self, **kwargs):

        # Start ROS
        ros_utils.start_master()

        # Launch EKF and aerial updater
        ros_utils.launch(package='localization', launch_file='jackal_ekf.launch')
        ros_utils.launch(package='localization', launch_file='aerial_global_updater.launch', argv={'ugv_poses_path': self.data_sources['ugv_poses_path']})

        # Start recording output bag
        output_bag_path = os.path.join(self.repetition_dir, '%s_output.bag' % self.name)
        self.results[self.repetition_id]['output_bag_path'] = output_bag_path
        ros_utils.start_recording_bag(output_bag_path, ['/odometry/filtered', '/odometry/filtered/aerial_updates']) # TODO: exceptions are thrown!!!

        # Start input bag and wait
        _, bag_duration = ros_utils.play_bag(self.data_sources['jackal_bag_path'], start_time=85) # TODO: read 60 from somewhere
        time.sleep(bag_duration)

        # Stop recording output bag
        ros_utils.stop_recording_bags()

        # Plot
        ekf = ros_utils.bag_to_dataframe(output_bag_path, topic=r'/odometry/filtered', fields=['pose.pose.position.x', 'pose.pose.position.y'])
        ekf_with_aerial_updates = ros_utils.bag_to_dataframe(output_bag_path, topic=r'/odometry/filtered/aerial_updates', fields=['pose.pose.position.x', 'pose.pose.position.y'])
        viz_utils.plot_2d_trajectory([ekf, ekf_with_aerial_updates], labels=['EKF', 'EKF with aerial updates'],
                                     file_name=os.path.join(self.repetition_dir, '2d_trajectory'))

if __name__ == '__main__':
    from content.data_pointers.lavi_november_18.jackal import jackal_18
    from content.data_pointers.lavi_november_18.dji import plot1_snapshots_80_meters_ugv_poses
    experiment = GlobalEkfUpdatesExperiment(name='global_ekf_updates',
                                            data_sources={'jackal_bag_path': jackal_18['10-25'].path, 'ugv_poses_path': plot1_snapshots_80_meters_ugv_poses},
                                            params={},
                                            working_dir=r'/home/omer/temp')
    experiment.run(repetitions=1)

    # TODO: think about the fact that input bag might overwrite TFs and topics related to the output