import os
import numpy as np
import pandas as pd

from experiments_framework.framework.experiment import Experiment
import experiments_framework.framework.utils as utils
import experiments_framework.framework.ros_utils as ros_utils
import experiments_framework.framework.viz_utils as viz_utils


class EkfExperiment(Experiment):

    def clean_env(self):
        pass

    def task(self, **kwargs):
        raw_ekf = ros_utils.bag_to_dataframe(self.data_sources, topic=r'/odometry/filtered', fields = ['pose.pose.position.x', 'pose.pose.position.y'])
        raw_ekf_duration = raw_ekf.index[-1] - raw_ekf.index[0]
        gps = ros_utils.bag_to_dataframe(self.data_sources, topic=r'/odometry/gps', fields = ['pose.pose.position.x', 'pose.pose.position.y'])
        gps_duration = gps.index[-1] - gps.index[0]
        raw_odom = ros_utils.bag_to_dataframe(self.data_sources, topic=r'/jackal_velocity_controller/odom', fields = ['pose.pose.position.x', 'pose.pose.position.y'])
        raw_odom_duration = raw_odom.index[-1] - raw_odom.index[0]
        ekf_with_gps = ros_utils.bag_to_dataframe(self.data_sources, topic=r'/odometry/filtered/global', fields = ['pose.pose.position.x', 'pose.pose.position.y'])
        ekf_with_gps_duration = ekf_with_gps.index[-1] - ekf_with_gps.index[0]

        avg_duration = np.mean([raw_ekf_duration, gps_duration, raw_odom_duration, ekf_with_gps_duration])

        max_xy = pd.concat([raw_ekf.max(), gps.max(), raw_odom.max(), ekf_with_gps.max()], axis=1).max(axis=1)
        min_xy = pd.concat([raw_ekf.min(), gps.min(), raw_odom.min(), ekf_with_gps.min()], axis=1).min(axis=1)

        for p in np.linspace(0.01, 1, num=20):
            p_raw_ekf = raw_ekf.head(int(p * len(raw_ekf.index)))
            p_gps = gps.head(int(p * len(gps.index)))
            p_raw_odom = raw_odom.head(int(p * len(raw_odom.index)))
            p_ekf_with_gps = ekf_with_gps.head(int(p * len(ekf_with_gps)))
            viz_utils.plot_2d_trajectory([p_raw_ekf, p_gps, p_raw_odom, p_ekf_with_gps], labels=['raw_ekf', 'gps', 'raw_odom', 'ekf_with_gps'],
                                         file_name=os.path.join(self.repetition_dir, '2d_trajectory_%d' % int(100*p)),
                                         xlim=(min_xy.loc['pose.pose.position.x']-15, max_xy.loc['pose.pose.position.x']+15),
                                         ylim=(min_xy.loc['pose.pose.position.y']-15, max_xy.loc['pose.pose.position.y']+15))


if __name__ == '__main__':

    import experiments_framework.content.data_pointers.lavi_april_18.jackal as jackal_data

    execution_dir = utils.create_new_execution_folder('ekf_jackal_18')
    for bag_name, bag_descriptor in jackal_data.forks:
        experiment = EkfExperiment(name='ekf on %s' % bag_name, data_sources=bag_descriptor.path, working_dir=execution_dir)
        experiment.run(repetitions=1)


    # TODO: plot also 3d trajectory