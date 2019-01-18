import os
import time

from framework import ros_utils
from framework import config

if __name__ == '__main__':

    ros_utils.start_master(use_sim_time=False)
    ros_utils.launch_rviz(os.path.join(config.root_dir_path, 'src/experiments_framework/framework/amcl.rviz'))

    # Launch base_link to contours_scan_link static TF
    ros_utils.launch(package='localization',
                     launch_file='static_identity_tf.launch',
                     argv={'frame_id': 'base_link', 'child_frame_id': 'contours_scan_link'})

    # Launch map server
    ros_utils.launch(package='localization',
                     launch_file='map.launch',
                     argv={'map_yaml_path': r'/home/omer/Downloads/panorama_cropped_map.yaml'})

    # Wait for map server to load
    ros_utils.wait_for_rosout_message(node_name='map_server',
                                      desired_message=r'Read a \d+ X \d+ map @ \d\.?\d+? m/cell',
                                      is_regex=True)

    # Launch AMCL
    ros_utils.launch(package='localization', launch_file='amcl.launch')

    # Wait for AMCL to load
    ros_utils.wait_for_rosout_message(node_name='amcl', desired_message='Done initializing likelihood field model.')

    # Launch ICP
    ros_utils.launch(package='localization', launch_file='icp.launch')

    # Start recording output bag
    ros_utils.start_recording_bag(r'/home/omer/Downloads/video_results.bag',
                                  ['/amcl_pose', '/particlecloud', '/scanmatcher_pose'])

    # Start video player and wait
    ros_utils.play_video_to_topic(r'/home/omer/orchards_ws/data/lavi_apr_18/raw/dji/DJI_0167.MP4',
                                  topic='/uav/camera/image_raw',
                                  frame_id='uav/camera_link')
    time.sleep(100000)

    # Stop recording output bag
    ros_utils.stop_recording_bags()

    ros_utils.kill_rviz()
    ros_utils.kill_master()

