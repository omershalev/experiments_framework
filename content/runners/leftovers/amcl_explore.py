import cv2
import numpy as np
import time

from experiments_framework.framework import ros_utils

if __name__ == '__main__':


    ros_utils.start_master()


    ros_utils.launch_rviz(r'/home/omer/.rviz/omer.rviz') # TODO: move the config to another location


    ros_utils.launch(package='localization', launch_file='static_identity_tf.launch',
                     argv={'frame_id': 'base_link', 'child_frame_id': 'contours_scan_link'})
    ros_utils.launch(package='localization', launch_file='synthetic_scan_generator.launch',
                     argv={'map_image_path': r'/home/omer/Downloads/temp_map_2.pgm'})
    ros_utils.launch(package='localization', launch_file='map.launch', argv={'map_yaml_file': r'/home/omer/Downloads/temp_map_2.yaml'})
    ros_utils.wait_for_rosout_message(node_name='map_server', desired_message=r'Read a \d+ X \d+ map @ \d\.?\d+? m/cell', is_regex=True)

    ros_utils.launch(package='localization', launch_file='amcl.launch')
    ros_utils.wait_for_rosout_message(node_name='amcl', desired_message='Done initializing likelihood field model.')


    ros_utils.launch(package='localization', launch_file='icp.launch')

    # ros_utils.wait_for_rosout_message(node_name='amcl', message_field='Global initialisation done!')
    # ros_utils.launch(package='localization', launch_file='static_identity_tf.launch',
    #                  argv={'frame_id': 'map', 'child_frame_id': 'odom'})


    _, bag_duration = ros_utils.play_bag('/home/omer/Downloads/19-03-5_simple_trajectory_17.bag')
    time.sleep(bag_duration)

    ros_utils.kill_rviz()

    ros_utils.kill_master()