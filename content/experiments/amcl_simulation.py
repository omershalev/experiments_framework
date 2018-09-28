import time
import os
import cv2
import numpy as np
import pandas as pd

from framework.experiment import Experiment
from framework import ros_utils
from framework import cv_utils
from framework import utils
from framework import config
from content.experiments.path_planning import PathPlanningExperiment
from computer_vision import maps_generation

class AmclSimulationExperiment(Experiment):

    def _launch_world_to_map_static_tf(self, namespace):
        ros_utils.launch(package='localization',
                         launch_file='static_identity_tf.launch',
                         argv={'ns': namespace, 'frame_id': 'world', 'child_frame_id': 'map', 'apply_ns_on_parent': False})

    def _launch_base_to_scan_static_tf(self, namespace):
        ros_utils.launch(package='localization',
                         launch_file='static_identity_tf.launch',
                         argv={'ns': namespace, 'frame_id': 'base_link', 'child_frame_id': 'contours_scan_link'})

    def _launch_synthetic_scan_generator(self, localization_image_path, namespace):
        ros_utils.launch(package='localization',
                         launch_file='synthetic_scan_generator.launch',
                         argv={'ns': namespace,
                               'localization_image_path': localization_image_path,
                               'min_angle': self.params['min_angle'],
                               'max_angle': self.params['max_angle'],
                               'samples_num': self.params['samples_num'],
                               'min_distance': self.params['min_distance'],
                               'max_distance': self.params['max_distance'],
                               'resolution': self.params['resolution'],
                               'r_primary_search_samples': self.params['r_primary_search_samples'],
                               'r_secondary_search_step': self.params['r_secondary_search_step'],
                               'scan_noise_sigma': self.params['scan_noise_sigma']}) # TODO: change hard coded value!!!!!!!!!!!!!

    def _launch_map_server(self, map_yaml_path, namespace):
        ros_utils.launch(package='localization',
                         launch_file='map.launch',
                         argv={'ns': namespace, 'map_yaml_path': map_yaml_path})
        ros_utils.wait_for_rosout_message(node_name='%s/map_server' % namespace,
                                          desired_message=r'Read a \d+ X \d+ map @ \d\.?\d+? m/cell',
                                          is_regex=True)

    def _launch_amcl(self, namespace):
        ros_utils.launch(package='localization', launch_file='amcl.launch', argv={'ns': namespace})
        ros_utils.wait_for_rosout_message(node_name='%s/amcl' % namespace, desired_message='Done initializing likelihood field model.')

    def _launch_synthetic_odometry(self, namespace):
        ros_utils.launch(package='localization', launch_file='synthetic_odometry.launch',
                         argv={'ns': namespace,
                               'resolution': self.params['resolution'],
                               'noise_mu_x': self.params['odometry_noise_mu_x'],
                               'noise_mu_y': self.params['odometry_noise_mu_y'],
                               'noise_sigma_x': self.params['odometry_noise_sigma_x'],
                               'noise_sigma_y': self.params['odometry_noise_sigma_y'],
                               })

    def _generate_results_dataframe(self, namespace, output_bag_path, image_height):
        ground_truth_df = ros_utils.bag_to_dataframe(output_bag_path, topic='/ugv_pose', fields=['x', 'y'])
        ground_truth_df['x'] = ground_truth_df['x'].apply(lambda cell: cell * config.top_view_resolution)
        ground_truth_df['y'] = ground_truth_df['y'].apply(lambda cell: (image_height - cell) * config.top_view_resolution)  # TODO: is this the height of the localization image??
        ground_truth_df.columns = ['ground_truth_x[%d]' % self.repetition_id, 'ground_truth_y[%d]' % self.repetition_id]
        amcl_pose_df = ros_utils.bag_to_dataframe(output_bag_path, topic='/%s/amcl_pose' % namespace, fields=['pose.pose.position.x', 'pose.pose.position.y'])
        amcl_pose_df.columns = ['amcl_pose_x[%d]' % (self.repetition_id), 'amcl_pose_y[%d]' % (self.repetition_id)]
        def covariance_norm(covariance_mat): # TODO: think about this with Amir
            return np.linalg.norm(np.array(covariance_mat).reshape(6, 6))
        amcl_covariance_df = ros_utils.bag_to_dataframe(output_bag_path, topic='/%s/amcl_pose' % namespace, fields=['pose.covariance'], aggregation=covariance_norm)
        amcl_covariance_df.columns = ['amcl_covariance_norm[%d]' % (self.repetition_id)]
        amcl_results_df = pd.concat([ground_truth_df, amcl_pose_df, amcl_covariance_df], axis=1)
        amcl_results_df['ground_truth_x[%d]' % self.repetition_id] = amcl_results_df['ground_truth_x[%d]' % self.repetition_id].interpolate()  # TODO: try method='time'
        amcl_results_df['ground_truth_y[%d]' % self.repetition_id] = amcl_results_df['ground_truth_y[%d]' % self.repetition_id].interpolate()  # TODO: try method='time'
        error = np.sqrt((amcl_results_df['ground_truth_x[%d]' % self.repetition_id] - amcl_results_df['amcl_pose_x[%d]' % (self.repetition_id)]) ** 2 + \
                        (amcl_results_df['ground_truth_y[%d]' % self.repetition_id] - amcl_results_df['amcl_pose_y[%d]' % (self.repetition_id)]) ** 2)
        amcl_results_df['amcl_pose_error[%d]' % (self.repetition_id)] = error
        amcl_results_path = os.path.join(self.repetition_dir, '%s_amcl_results.csv' % namespace)
        amcl_results_df.to_csv(amcl_results_path)
        self.results[self.repetition_id]['%s_amcl_results_path' % namespace] = amcl_results_path

    def clean_env(self):
        utils.kill_process('amcl')
        ros_utils.kill_master()

    def task(self, **kwargs):

        launch_rviz = kwargs.get('launch_rviz', False)

        origin_map_image_path = self.data_sources['map_image_path']
        semantic_trunks = self.data_sources['semantic_trunks'] # TODO: this will become: map semantic_trunks (localization_semantic_trunks will be added and used for alignment)
        origin_localization_image_path = self.data_sources['localization_image_path']
        gaussian_scale_factor = self.params['gaussian_scale_factor']
        bounding_box_expand_ratio = self.params['bounding_box_expand_ratio']
        mean_trunk_radius = self.params['mean_trunk_radius']
        std_trunk_radius = self.params['std_trunk_radius']

        # Generate canopies and trunk map images
        map_image = cv2.imread(origin_map_image_path)
        canopies_map_image = maps_generation.generate_canopies_map(map_image)
        trunks_map_image = maps_generation.generate_trunks_map(map_image, semantic_trunks.values(),
                                                               mean_trunk_radius, std_trunk_radius, np_random_state=self.np_random_state) # TODO: verify that std is not too small!!!
        upper_left, lower_right = cv_utils.get_bounding_box(canopies_map_image, semantic_trunks.values(), expand_ratio=bounding_box_expand_ratio)
        canopies_map_image = canopies_map_image[upper_left[1]:lower_right[1], upper_left[0]:lower_right[0]]
        trunks_map_image = trunks_map_image[upper_left[1]:lower_right[1], upper_left[0]:lower_right[0]]
        canopies_map_yaml_path, _ = ros_utils.save_image_to_map(canopies_map_image, resolution=self.params['resolution'], map_name='canopies_map', dir_name=self.repetition_dir)
        trunks_map_yaml_path, _ = ros_utils.save_image_to_map(trunks_map_image, resolution=self.params['resolution'], map_name='trunks_map', dir_name=self.repetition_dir)

        # Generate canopies and trunks localization images
        localization_image = cv2.imread(origin_localization_image_path)
        # TODO: with two images, localization_image will have to be warped to map_image here (and only then cropped, according to the following line)
        canopies_localization_image = maps_generation.generate_canopies_map(localization_image)
        trunks_localization_image = maps_generation.generate_trunks_map(map_image, semantic_trunks.values(),
                                                                        mean_trunk_radius, std_trunk_radius, np_random_state=self.np_random_state)
        canopies_localization_image = canopies_localization_image[upper_left[1]:lower_right[1], upper_left[0]:lower_right[0]]
        trunks_localization_image = trunks_localization_image[upper_left[1]:lower_right[1], upper_left[0]:lower_right[0]]
        canopies_localization_image_path = os.path.join(self.repetition_dir, 'canopies_localization.jpg')
        cv2.imwrite(canopies_localization_image_path, canopies_localization_image)
        trunks_localization_image_path = os.path.join(self.repetition_dir, 'trunks_localization.jpg')
        cv2.imwrite(trunks_localization_image_path, trunks_localization_image)

        ##### WIP #####
        wp1 = tuple(np.array(semantic_trunks['7/A']) + np.array([mean_trunk_radius + 70, 0]))
        wp2 = tuple(np.array(semantic_trunks['7/G']) + np.array([mean_trunk_radius + 115, 0]))
        wp3 = tuple(np.array(semantic_trunks['7/A']) + np.array([mean_trunk_radius + 70, 0]))
        wp4 = tuple(np.array(semantic_trunks['6/A']) + np.array([mean_trunk_radius + 70, 0]))
        wp5 = tuple(np.array(semantic_trunks['6/G']) + np.array([mean_trunk_radius + 115, 0]))
        wp6 = tuple(np.array(semantic_trunks['6/A']) + np.array([mean_trunk_radius + 70, 0]))
        # waypoints = [wp1, wp2, wp3, wp4, wp5, wp6] # TODO: rearrange
        waypoints = [wp1, wp2] # TODO: rearrange
        optimized_sigma = 90.39 # TODO: rearrange
        freq = 30 # TODO: rearrange and consider what frequency to use!!!
        experiment = PathPlanningExperiment(name='path_planning',
                                            data_sources={'map_image_path': origin_map_image_path, 'trunk_points_list': semantic_trunks.values(),
                                                          'map_upper_left': upper_left, 'map_lower_right': lower_right, 'waypoints': waypoints},
                                            params={'trunk_radius': mean_trunk_radius, 'gaussian_scale_factor': gaussian_scale_factor,
                                                    'canopy_sigma': optimized_sigma, 'bounding_box_expand_ratio': bounding_box_expand_ratio},
                                            working_dir=self.repetition_dir, metadata=self.metadata)
        experiment.run(repetitions=1)
        trajectory = experiment.results[1]['trajectory']
        timestamps = np.linspace(start=0, stop=(1.0 / freq) * len(trajectory), num=len(trajectory))
        pose_time_tuples_list = [(x, y, t) for (x, y), t in zip(trajectory, timestamps)]
        trajectory_bag_path = os.path.join(self.repetition_dir, 'trajectory.bag')
        ros_utils.trajectory_to_bag(pose_time_tuples_list, trajectory_bag_path)
        ##### E/O WIP #####

        # Start ROS and RVIZ
        ros_utils.start_master()
        if launch_rviz:
            ros_utils.launch_rviz(os.path.join(config.root_dir_path, 'src/experiments_framework/framework/amcl.rviz'))

        # Launch localization stack for canopies and trunks
        self._launch_world_to_map_static_tf(namespace='canopies')
        self._launch_world_to_map_static_tf(namespace='trunks')
        self._launch_base_to_scan_static_tf(namespace='canopies')
        self._launch_base_to_scan_static_tf(namespace='trunks')
        self._launch_synthetic_scan_generator(canopies_localization_image_path, namespace='canopies')
        self._launch_synthetic_scan_generator(trunks_localization_image_path, namespace='trunks') # TODO: adjust parameters to increase trunks scan FPS (it is currently 15!!!)
        self._launch_map_server(canopies_map_yaml_path, namespace='canopies')
        self._launch_map_server(trunks_map_yaml_path, namespace='trunks')
        self._launch_amcl(namespace='canopies')
        self._launch_amcl(namespace='trunks')
        self._launch_synthetic_odometry(namespace='canopies')
        self._launch_synthetic_odometry(namespace='trunks')

        # Start recording output bag
        output_bag_path = os.path.join(self.repetition_dir, '%s_output.bag' % self.name)
        self.results[self.repetition_id]['output_bag_path'] = output_bag_path
        ros_utils.start_recording_bag(output_bag_path, ['/ugv_pose', '/canopies/amcl_pose', '/canopies/particlecloud',
                                                        '/trunks/amcl_pose', '/trunks/particlecloud']) # TODO: exceptions are thrown!!!

        # Start input bag and wait
        _, bag_duration = ros_utils.play_bag(trajectory_bag_path)
        time.sleep(bag_duration)

        # Stop recording output bag
        ros_utils.stop_recording_bags()

        # Kill RVIZ
        if launch_rviz:
            ros_utils.kill_rviz()

        # Generate results dafaframe
        self._generate_results_dataframe(namespace='canopies', output_bag_path=output_bag_path, image_height=canopies_localization_image.shape[0])
        self._generate_results_dataframe(namespace='trunks', output_bag_path=output_bag_path, image_height=trunks_localization_image.shape[0])