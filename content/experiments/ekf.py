import os
import rosbag

from experiments_framework.framework.experiment import Experiment
import experiments_framework.framework.utils as utils
import experiments_framework.framework.ros_utils as ros_utils
import experiments_framework.framework.viz_utils as viz_utils


class EkfExperiment(Experiment):

    def clean_env(self):
        pass

    def task(self, **kwargs):
        raw_ekf = ros_utils.bag_to_dataframe(self.data_sources, topic=r'/odometry/filtered', fields = ['pose.pose.position.x', 'pose.pose.position.y'])
        gps = ros_utils.bag_to_dataframe(self.data_sources, topic=r'/odometry/gps', fields = ['pose.pose.position.x', 'pose.pose.position.y'])
        raw_odom = ros_utils.bag_to_dataframe(self.data_sources, topic=r'/jackal_velocity_controller/odom', fields = ['pose.pose.position.x', 'pose.pose.position.y'])
        ekf_with_gps = ros_utils.bag_to_dataframe(self.data_sources, topic=r'/odometry/filtered/global', fields = ['pose.pose.position.x', 'pose.pose.position.y'])

        viz_utils.plot_2d_trajectory([raw_ekf, gps, raw_odom, ekf_with_gps], labels=['raw_ekf', 'gps', 'raw_odom', 'ekf_with_gps'], dir_name=self.repetition_dir)


if __name__ == '__main__':

    import experiments_framework.content.data_pointers.lavi_april_18.jackal as jackal_data

    execution_dir = utils.create_new_execution_folder('ekf_jackal_18')
    for bag_name, bag_descriptor in jackal_data.forks:
        experiment = EkfExperiment(name='ekf on %s' % bag_name, data_sources=bag_descriptor.path, working_dir=execution_dir)
        experiment.run(repetitions=1)