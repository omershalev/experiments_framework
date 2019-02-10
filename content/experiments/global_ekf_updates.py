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
        ros_utils.launch(package='localization', launch_file='aerial_global_updater.launch',
                         argv={'ugv_poses_path': self.data_sources['ugv_poses_path'],
                               'relevant_update_index': self.data_sources['relevant_update_index'],
                               'resolution': self.params['resolution']})

        # Start recording output bag
        output_bag_path = os.path.join(self.repetition_dir, '%s_output.bag' % self.name)
        ros_utils.start_recording_bag(output_bag_path, ['/odometry/filtered', '/odometry/filtered/aerial_updates'])

        # Start input bag and wait
        _, bag_duration = ros_utils.play_bag(self.data_sources['jackal_bag_path'], start_time=self.params['bag_start_time'])
        time.sleep(bag_duration)

        # Stop recording output bag
        ros_utils.stop_recording_bags()

        # Plot
        ekf_df = ros_utils.bag_to_dataframe(output_bag_path, topic=r'/odometry/filtered', fields=['pose.pose.position.x', 'pose.pose.position.y'])
        ekf_with_aerial_update_df = ros_utils.bag_to_dataframe(output_bag_path, topic=r'/odometry/filtered/aerial_updates',
                                                               fields=['pose.pose.position.x', 'pose.pose.position.y'])
        ekf_df.to_csv(os.path.join(self.repetition_dir, 'ekf_pose.csv'))
        ekf_with_aerial_update_df.to_csv(os.path.join(self.repetition_dir, 'ekf_with_aerial_update_df_pose.csv'))
        self.results[self.repetition_id]['ekf_pose_path'] = os.path.join(self.repetition_dir, 'ekf_pose.csv')
        self.results[self.repetition_id]['ekf_with_aerial_update_path'] = os.path.join(self.repetition_dir, 'ekf_with_aerial_update_pose.csv')
        viz_utils.plot_2d_trajectory([ekf_df, ekf_with_aerial_update_df], labels=['EKF', 'EKF with aerial updates'],
                                     file_name=os.path.join(self.repetition_dir, '2d_trajectory'))

        # Delete bag file
        os.remove(output_bag_path)


    # TODO: think about the fact that input bag might overwrite TFs and topics related to the output