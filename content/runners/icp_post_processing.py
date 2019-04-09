import os
import pandas as pd
import matplotlib.pyplot as plt

from framework import utils
from framework import viz_utils

#################################################################################################
#                                             CONFIG                                            #
#################################################################################################
original_experiment_dir = r'/home/omer/temp/20190316-152308_apr_icp/20190316-152308_icp_snapshots_for_fork_trajectory_on_15-08-1'
repetition_ids = [1, 2, 3]
#################################################################################################


if __name__ == '__main__':
    execution_dir = utils.create_new_execution_folder('icp_post_processing')
    for repetition_id in repetition_ids:
        plt.figure()
        icp_df = pd.read_csv(os.path.join(original_experiment_dir, str(repetition_id), 'canopies_icp_results.csv'), index_col=0)
        viz_utils.plot_2d_trajectory([icp_df.loc[:, ['icp_pose_x[%d]' % repetition_id, 'icp_pose_y[%d]' % repetition_id]]])
        plt.savefig(os.path.join(execution_dir, 'icp_%d.jpg' % repetition_id))
