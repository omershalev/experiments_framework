import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from framework import utils
from framework import viz_utils

#################################################################################################
#                                             CONFIG                                            #
#################################################################################################
original_execution_dir = r'/home/omer/orchards_ws/results/global_ekf_updates'
#################################################################################################


if __name__ == '__main__':
    execution_dir = utils.create_new_execution_folder('global_ekf_updates_post_processing')
    ekf_errors = {}
    ekf_with_aerial_update_errors = {}
    fig = plt.figure(figsize=(8.5, 11))
    fig.tight_layout()
    for experiment_idx, experiment_name in enumerate(os.listdir(original_execution_dir)):
        with open(os.path.join(original_execution_dir, experiment_name, 'experiment_summary.json')) as f:
            experiment_summary = json.load(f)
        with open(experiment_summary['data_sources']['ugv_poses_path']) as f:
            ugv_poses_path = json.load(f)
        relevant_update_index = experiment_summary['data_sources']['relevant_update_index']
        ekf_df = pd.read_csv(experiment_summary['results']['1']['ekf_pose_path'], index_col=0)
        ekf_with_aerial_update_df = pd.read_csv(experiment_summary['results']['1']['ekf_with_aerial_update_path'], index_col=0)
        update_time = ekf_with_aerial_update_df.loc[(ekf_with_aerial_update_df['pose.pose.position.x'].diff().abs() > 0.5) |
                                                    (ekf_with_aerial_update_df['pose.pose.position.y'].diff().abs() > 0.5)].index[0]
        ekf_before_update_df = ekf_with_aerial_update_df[ekf_with_aerial_update_df.index < update_time]
        ekf_without_update_df = ekf_df[ekf_df.index > update_time]
        ekf_with_update_df = ekf_with_aerial_update_df[ekf_with_aerial_update_df.index > update_time]
        update_arrow_df = pd.concat([ekf_before_update_df.iloc[-1], ekf_with_update_df.iloc[0]], axis=1).transpose()
        fig.add_subplot(3, 3, experiment_idx + 1)
        label_x_axis = True if experiment_idx in [6, 7, 8] else False
        label_y_axis = True if experiment_idx in [0, 3, 6] else False
        viz_utils.plot_2d_trajectory([ekf_before_update_df, ekf_without_update_df, ekf_with_update_df],
                                     colors=['black', 'black', 'deeppink'], label_x_axis=label_x_axis, label_y_axis=label_y_axis)
        plt.plot(update_arrow_df.iloc[:,0], update_arrow_df.iloc[:,1], 'deeppink', linestyle='--')
        ax = plt.gca()
        start_circle = plt.Circle(tuple(ekf_before_update_df.iloc[0]), 0.5, color='black')
        update_circle = plt.Circle(tuple(ekf_before_update_df.iloc[-1]), 1.2, color='lime')
        end_with_update_circle = plt.Circle(tuple(ekf_with_update_df.iloc[-1]), 0.5, color='deeppink')
        end_without_update_circle = plt.Circle(tuple(ekf_without_update_df.iloc[-1]), 0.5, color='black')
        ax.add_artist(start_circle)
        ax.add_artist(update_circle)
        ax.add_artist(end_with_update_circle)
        ax.add_artist(end_without_update_circle)
        plt.title(experiment_idx)
        ekf_errors[relevant_update_index] = np.linalg.norm(ekf_df.iloc[-1])
        ekf_with_aerial_update_errors[relevant_update_index] = np.linalg.norm(ekf_with_aerial_update_df.iloc[-1])
    errors_comparison_df = pd.concat([pd.Series(ekf_errors), pd.Series(ekf_with_aerial_update_errors)], axis=1).rename(columns={0: 'baseline', 1: 'aerial_updates'})
    errors_comparison_df.to_csv(os.path.join(execution_dir, 'errors.csv'))
    plt.savefig(os.path.join(execution_dir, 'ekf_updates.jpg'))
