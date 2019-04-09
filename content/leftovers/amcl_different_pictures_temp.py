import os
import time

from experiments_framework.framework import ros_utils
from experiments_framework.framework import config

if __name__ == '__main__':

    ros_utils.start_master()
    ros_utils.launch_rviz(os.path.join(config.root_dir_path, 'astar/experiments_framework/framework/amcl.rviz'))

    # Launch base_link to contours_scan_link static TF
    ros_utils.launch(package='localization',
                     launch_file='static_identity_tf.launch',
                     argv={'frame_id': 'base_link', 'child_frame_id': 'contours_scan_link'})

    # Launch synthetic scan generator
    ros_utils.launch(package='localization',
                     launch_file='synthetic_scan_generator.launch',
                     argv={'map_image_path': r'/home/omer/Downloads/dji_15-53-1_map.pgm'})

    # Launch map server
    ros_utils.launch(package='localization',
                     launch_file='map.launch',
                     argv={'map_yaml_file': r'/home/omer/Downloads/dji_15-08-1_map.yaml'})

    # Wait for map server to load
    ros_utils.wait_for_rosout_message(node_name='map_server',
                                      desired_message=r'Read a \d+ X \d+ map @ \d\.?\d+? m/cell',
                                      is_regex=True)

    # Launch AMCL
    ros_utils.launch(package='localization', launch_file='amcl.launch')

    # Wait for AMCL to load
    ros_utils.wait_for_rosout_message(node_name='amcl', desired_message='Done initializing likelihood field model.')

    # Launch ICP
    ros_utils.launch(package='localization', launch_file='scan_matcher.launch')

    # Start bag file and wait
    _, bag_duration = ros_utils.play_bag('/home/omer/Downloads/15-53-1_simple_trajectory_2.bag')
    time.sleep(bag_duration)

    ros_utils.kill_rviz()
    ros_utils.kill_master()