import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from framework.experiment import Experiment
import framework.ros_utils as ros_utils
import framework.viz_utils as viz_utils


class Ekf(Experiment):

    def clean_env(self):
        pass

    def task(self, **kwargs):

        maxima = []
        minima = []
        plot_dict = {}
        if kwargs.get('odom'):
            odom = ros_utils.bag_to_dataframe(self.data_sources, topic=r'/jackal_velocity_controller/odom', fields = ['pose.pose.position.x', 'pose.pose.position.y'])
            maxima.append(odom.max())
            minima.append(odom.min())
            plot_dict['odom'] = odom
            self.results[self.repetition_id]['odom_error'] = np.linalg.norm(odom.iloc[-1] - odom.iloc[0])
        if kwargs.get('ekf'):
            ekf = ros_utils.bag_to_dataframe(self.data_sources, topic=r'/odometry/filtered', fields = ['pose.pose.position.x', 'pose.pose.position.y'])
            maxima.append((ekf.max()))
            minima.append((ekf.min()))
            plot_dict['ekf'] = ekf
            self.results[self.repetition_id]['ekf_error'] = np.linalg.norm(ekf.iloc[-1] - ekf.iloc[0])
        if kwargs.get('gps'):
            gps = ros_utils.bag_to_dataframe(self.data_sources, topic=r'/odometry/filtered/global', fields=['pose.pose.position.x', 'pose.pose.position.y'])
            maxima.append(gps.max())
            minima.append(gps.min())
            plot_dict['gps'] = gps
            self.results[self.repetition_id]['gps_error'] = np.linalg.norm(gps.iloc[-1] - gps.iloc[0])

        max_xy = pd.concat(maxima, axis=1).max(axis=1)
        min_xy = pd.concat(minima, axis=1).min(axis=1)

        plots_colormap = {'odom': 'r', 'ekf': 'g', 'gps': 'b'}
        for p in np.linspace(0.01, 1, num=20):
            plot_list = []
            colors = []
            for plot_name, df in plot_dict.items():
                plot_list.append(df.head(int(p * len(df.index))))
                colors.append(plots_colormap[plot_name])
            plt.figure()
            viz_utils.plot_2d_trajectory(plot_list, labels=plot_dict.keys(), colors=colors,
                                         file_name=os.path.join(self.repetition_dir, '2d_trajectory_%d' % int(100*p)),
                                         xlim=(min_xy.loc['pose.pose.position.x']-15, max_xy.loc['pose.pose.position.x']+15),
                                         ylim=(min_xy.loc['pose.pose.position.y']-15, max_xy.loc['pose.pose.position.y']+15))