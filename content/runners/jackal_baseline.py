import os
import matplotlib.pyplot as plt

from framework import utils
from framework import ros_utils
from framework import viz_utils
from framework import config

#################################################################################################
#                                             CONFIG                                            #
#################################################################################################
ugv_bag_name = '18-24-26'
icp_experiment_path = os.path.join(config.base_results_path, 'jackal_icp')
setup = 'apr' # apr / nov / lab
#################################################################################################

if setup == 'apr':
    from content.data_pointers.lavi_april_18.jackal import jackal_18 as ugv_pointers
elif setup == 'nov':
    raise NotImplementedError
elif setup == 'lab':
    raise NotImplementedError

if __name__ == '__main__':

    execution_dir = utils.create_new_execution_folder('jackal_baseline')

    bag_descriptor = ugv_pointers[ugv_bag_name]
    odom = ros_utils.bag_to_dataframe(bag_descriptor.path, topic=r'/jackal_velocity_controller/odom', fields=['pose.pose.position.x', 'pose.pose.position.y'])
    ekf = ros_utils.bag_to_dataframe(bag_descriptor.path, topic=r'/odometry/filtered', fields=['pose.pose.position.x', 'pose.pose.position.y'])
    icp_bag_path = os.path.join(icp_experiment_path, '1', 'icp_on_apr_%s_output.bag' % ugv_bag_name)
    icp = ros_utils.bag_to_dataframe(icp_bag_path, topic=r'/scanmatcher_pose', fields=['x', 'y'])

    plt.figure()
    viz_utils.plot_2d_trajectory([odom])
    plt.title('raw odometry')
    plt.tight_layout()
    plt.savefig(os.path.join(execution_dir, 'raw_odometry.jpg'))

    plt.figure()
    viz_utils.plot_2d_trajectory([ekf])
    plt.title('EKF')
    plt.tight_layout()
    plt.savefig(os.path.join(execution_dir, 'ekf.jpg'))

    plt.figure()
    viz_utils.plot_2d_trajectory([icp])
    plt.title('ICP')
    plt.tight_layout()
    plt.savefig(os.path.join(execution_dir, 'icp.jpg'))