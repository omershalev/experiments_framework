import time
import json
import os
import cv2

from framework.experiment import Experiment
from framework import ros_utils
from framework import cv_utils
from framework import utils
from framework import config
from computer_vision import maps_generation
from computer_vision import calibration


class AmclVideoExperiment(Experiment):

    def _launch_base_to_scan_static_tf(self, namespace):
        ros_utils.launch(package='localization',
                         launch_file='static_identity_tf.launch',
                         argv={'ns': namespace, 'frame_id': 'base_link', 'child_frame_id': 'contours_scan_link'})

    def _launch_map_server(self, map_yaml_path, namespace):
        ros_utils.launch(package='localization',
                         launch_file='map.launch',
                         argv={'ns': namespace, 'map_yaml_path': map_yaml_path})
        ros_utils.wait_for_rosout_message(node_name='%s/map_server' % namespace,
                                          desired_message=r'Read a \d+ X \d+ map @ \d\.?\d+? m/cell',
                                          is_regex=True)

    def _launch_amcl(self, namespace):
        ros_utils.launch(package='localization', launch_file='amcl.launch', argv={'ns': namespace, 'min_particles': self.params['min_amcl_particles']})
        ros_utils.wait_for_rosout_message(node_name='%s/amcl' % namespace, desired_message='Done initializing likelihood field model.')

    def _initialize_global_localization(self, namespace):
        ros_utils.wait_for_rosout_message(node_name='%s/amcl' % namespace, desired_message='Done initializing likelihood field model.')
        ros_utils.launch(package='localization', launch_file='global_localization_init.launch', argv={'ns': namespace})

    def clean_env(self):
        utils.kill_process('amcl')
        ros_utils.kill_master()

    def task(self, **kwargs):

        launch_rviz = kwargs.get('launch_rviz', False)

        map_image_path = self.data_sources['map_image_path']
        map_semantic_trunks = self.data_sources['map_semantic_trunks']
        scans_and_poses_pickle_path = self.data_sources['scans_and_poses_pickle_path']
        odom_pickle_path = self.data_sources['odom_pickle_path']
        video_path = self.data_sources['video_path']
        bounding_box_expand_ratio = self.params['bounding_box_expand_ratio']
        map_resolution = self.params['map_resolution']

        # Generate canopies map image
        map_image = cv2.imread(map_image_path)
        cv2.imwrite(os.path.join(self.experiment_dir, 'image_for_map.jpg'), map_image)
        canopies_map_image = maps_generation.generate_canopies_map(map_image)
        upper_left, lower_right = cv_utils.get_bounding_box(canopies_map_image, map_semantic_trunks.values(), expand_ratio=bounding_box_expand_ratio)
        canopies_map_image = canopies_map_image[upper_left[1]:lower_right[1], upper_left[0]:lower_right[0]]
        canopies_map_yaml_path, _ = ros_utils.save_image_to_map(canopies_map_image, resolution=map_resolution,
                                                                map_name='canopies_map', dir_name=self.experiment_dir)

        # Start ROS and RVIZ
        ros_utils.start_master(use_sim_time=False)
        if launch_rviz:
            ros_utils.launch_rviz(os.path.join(config.root_dir_path, 'src/experiments_framework/framework/amcl_video.rviz'))

        # Launch localization stack for canopies
        self._launch_base_to_scan_static_tf(namespace='canopies')
        self._launch_map_server(canopies_map_yaml_path, namespace='canopies')
        self._launch_amcl(namespace='canopies')
        self._initialize_global_localization(namespace='canopies')

        # Start playback
        ros_utils.launch(package='localization',
                         launch_file='scan_pose_odom_playback.launch',
                         argv={'ns': 'canopies',
                               'scans_and_poses_pickle_path': scans_and_poses_pickle_path,
                               'odom_pickle_path': odom_pickle_path,
                               'video_path': video_path})

        time.sleep(500) # TODO: remove

        # Stop recording output bag
        # ros_utils.stop_recording_bags()

        # Kill RVIZ
        if launch_rviz:
            ros_utils.kill_rviz()


if __name__ == '__main__':

    from content.data_pointers.lavi_april_18.dji import trunks_detection_results_dir as td_results_dir
    from content.data_pointers.lavi_april_18 import base_resources_path, base_raw_data_path
    from content.data_pointers.lavi_april_18 import orchard_topology

    execution_dir = utils.create_new_execution_folder('amcl_video')

    td_exp_name_for_map = 'manual_apr_15-08-1'
    with open(os.path.join(td_results_dir, td_exp_name_for_map, 'experiment_summary.json')) as f:
        td_summary_for_map = json.load(f)
    image_path = td_summary_for_map['data_sources']
    map_semantic_trunks = td_summary_for_map['results']['0']['semantic_trunks']
    pixel_to_meter_ratio_for_map = calibration.calculate_pixel_to_meter(td_summary_for_map['results']['0']['optimized_grid_dim_x'],
                                                                        td_summary_for_map['results']['0']['optimized_grid_dim_y'],
                                                                        orchard_topology.measured_row_widths,
                                                                        orchard_topology.measured_intra_row_distances)
    map_resolution = 1.0 / pixel_to_meter_ratio_for_map
    experiment = AmclVideoExperiment(name='amcl_video',
                                     data_sources={'map_image_path': image_path, 'map_semantic_trunks': map_semantic_trunks,
                                                   'scans_and_poses_pickle_path': os.path.join(base_resources_path, 'amcl_video', 'scans_and_poses.pkl'),
                                                   'odom_pickle_path': os.path.join(base_resources_path, 'amcl_video', 'ugv_odometry.pkl'),
                                                   'video_path':os.path.join(base_raw_data_path, 'dji', 'DJI_0167.MP4')},
                                     params={'bounding_box_expand_ratio': config.bounding_box_expand_ratio, 'map_resolution': map_resolution,
                                             'min_amcl_particles': 2000},
                                     working_dir=execution_dir)
    experiment.run(repetitions=1, launch_rviz=True)