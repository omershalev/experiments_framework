import time
import pickle
import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from framework.experiment import Experiment
from framework import ros_utils
from framework import cv_utils
from framework import viz_utils
from framework import utils
from framework import config
from content.experiments.path_planning import PathPlanningExperiment
from computer_vision import maps_generation
from computer_vision import offline_synthetic_scan_generator

class AmclSimulationExperiment(Experiment):

    def _launch_world_to_map_static_tf(self, namespace):
        ros_utils.launch(package='localization',
                         launch_file='static_identity_tf.launch',
                         argv={'ns': namespace, 'frame_id': 'world', 'child_frame_id': 'map', 'apply_ns_on_parent': False})

    def _launch_base_to_scan_static_tf(self, namespace):
        ros_utils.launch(package='localization',
                         launch_file='static_identity_tf.launch',
                         argv={'ns': namespace, 'frame_id': 'base_link', 'child_frame_id': 'contours_scan_link'})

    def _launch_synthetic_scan_generator(self, localization_image_path, namespace, pickle_path=None):
        ros_utils.launch(package='localization',
                         launch_file='synthetic_scan_generator.launch',
                         argv={'ns': namespace,
                               'localization_image_path': localization_image_path,
                               'min_angle': self.params['min_angle'],
                               'max_angle': self.params['max_angle'],
                               'samples_num': self.params['samples_num'],
                               'min_distance': self.params['min_distance'],
                               'max_distance': self.params['max_distance'],
                               'resolution': self.params['localization_resolution'],
                               'r_primary_search_samples': self.params['r_primary_search_samples'],
                               'r_secondary_search_step': self.params['r_secondary_search_step'],
                               'scan_noise_sigma': self.params['scan_noise_sigma'],
                               'scans_pickle_path': pickle_path})

    def _launch_map_server(self, map_yaml_path, namespace):
        ros_utils.launch(package='localization',
                         launch_file='map.launch',
                         argv={'ns': namespace, 'map_yaml_path': map_yaml_path})
        ros_utils.wait_for_rosout_message(node_name='%s/map_server' % namespace,
                                          desired_message=r'Read a \d+ X \d+ map @ \d\.?\d+? m/cell',
                                          is_regex=True)

    def _launch_amcl(self, namespace):
        ros_utils.launch(package='localization', launch_file='amcl.launch', argv={'ns': namespace, 'min_particles': self.params['min_amcl_particles']})

    def _initialize_global_localization(self, namespace):
        ros_utils.wait_for_rosout_message(node_name='%s/amcl' % namespace, desired_message='Done initializing likelihood field model.')
        ros_utils.launch(package='localization', launch_file='global_localization_init.launch', argv={'ns': namespace})

    def _launch_synthetic_odometry(self, namespace):
        ros_utils.launch(package='localization', launch_file='synthetic_odometry.launch',
                         argv={'ns': namespace,
                               'resolution': self.params['localization_resolution'],
                               'noise_mu_x': self.params['odometry_noise_mu_x'],
                               'noise_mu_y': self.params['odometry_noise_mu_y'],
                               'noise_sigma_x': self.params['odometry_noise_sigma_x'],
                               'noise_sigma_y': self.params['odometry_noise_sigma_y'],
                               })

    def _launch_contours_visualization(self):
        ros_utils.launch(package='localization',
                         launch_file='contours_scan_visualizer.launch',
                         argv={'min_angle': self.params['min_angle'],
                               'max_angle': self.params['max_angle'],
                               'resolution': self.params['localization_resolution'],
                               'window_size': int(self.params['max_distance'] * 1.3),
                               'canopies_image_path': self.viz_image_path,
                               'trunks_image_path': self.trunks_localization_image_path,
                               'canopies_scans_pickle_path': self.canopies_scans_pickle_path,
                               'trunks_scans_pickle_path': self.trunks_scans_pickle_path})

    def _generate_results_dataframe(self, namespace, output_bag_path, image_height):
        ground_truth_df = ros_utils.bag_to_dataframe(output_bag_path, topic='/ugv_pose', fields=['point.x', 'point.y'])
        ground_truth_df['point.x'] = ground_truth_df['point.x'].apply(lambda cell: cell * self.params['localization_resolution'])
        ground_truth_df['point.y'] = ground_truth_df['point.y'].apply(lambda cell: (image_height - cell) * self.params['localization_resolution'])  # TODO: is this the height of the localization image??
        ground_truth_df.columns = ['ground_truth_x[%d]' % self.repetition_id, 'ground_truth_y[%d]' % self.repetition_id]
        amcl_pose_df = ros_utils.bag_to_dataframe(output_bag_path, topic='/%s/amcl_pose' % namespace, fields=['pose.pose.position.x', 'pose.pose.position.y'])
        amcl_pose_df.columns = ['amcl_pose_x[%d]' % (self.repetition_id), 'amcl_pose_y[%d]' % (self.repetition_id)]
        def covariance_norm(covariance_mat): # TODO: think about this with Amir
            return np.linalg.norm(np.array(covariance_mat).reshape(6, 6))
        amcl_covariance_df = ros_utils.bag_to_dataframe(output_bag_path, topic='/%s/amcl_pose' % namespace, fields=['pose.covariance'], aggregation=covariance_norm)
        amcl_covariance_df.columns = ['amcl_covariance_norm[%d]' % (self.repetition_id)]
        ground_truth_df = ground_truth_df[~ground_truth_df.index.duplicated(keep='first')] # TODO: new logic, suspect it!
        amcl_pose_df = amcl_pose_df[~amcl_pose_df.index.duplicated(keep='first')] # TODO: new logic, suspect it!
        amcl_covariance_df = amcl_covariance_df[~amcl_covariance_df.index.duplicated(keep='first')] # TODO: new logic, suspect it!

        amcl_results_df = pd.concat([ground_truth_df, amcl_pose_df, amcl_covariance_df], axis=1)
        amcl_results_df['ground_truth_x[%d]' % self.repetition_id] = amcl_results_df['ground_truth_x[%d]' % self.repetition_id].interpolate()  # TODO: try method='time'
        amcl_results_df['ground_truth_y[%d]' % self.repetition_id] = amcl_results_df['ground_truth_y[%d]' % self.repetition_id].interpolate()  # TODO: try method='time'
        error = np.sqrt((amcl_results_df['ground_truth_x[%d]' % self.repetition_id] - amcl_results_df['amcl_pose_x[%d]' % (self.repetition_id)]) ** 2 + \
                        (amcl_results_df['ground_truth_y[%d]' % self.repetition_id] - amcl_results_df['amcl_pose_y[%d]' % (self.repetition_id)]) ** 2)
        amcl_results_df['amcl_pose_error[%d]' % (self.repetition_id)] = error
        amcl_results_path = os.path.join(self.repetition_dir, '%s_amcl_results.csv' % namespace)
        amcl_results_df.to_csv(amcl_results_path)
        self.results[self.repetition_id]['%s_amcl_results_path' % namespace] = amcl_results_path

    def _get_errors_and_covariance_norms_dataframes(self, namespace):
        joint_amcl_results_df = pd.DataFrame()
        for repetition_id in self.valid_repetitions:
            amcl_results_df = pd.read_csv(self.results[repetition_id]['%s_amcl_results_path' % namespace], index_col=0)
            joint_amcl_results_df = pd.concat([joint_amcl_results_df, amcl_results_df], axis=1)
        joint_amcl_results_df = joint_amcl_results_df.fillna(method='ffill')
        errors_df = joint_amcl_results_df[['amcl_pose_error[%d]' % repetition_id for repetition_id in self.valid_repetitions]]
        covariance_norms_df = joint_amcl_results_df[['amcl_covariance_norm[%d]' % repetition_id for repetition_id in self.valid_repetitions]]
        return errors_df, covariance_norms_df

    def _plot_canopies_vs_trunks(self, plot_name, canopies_vector, trunks_vector, canopies_stds, trunks_stds, output_dir):
        plt.figure()
        viz_utils.plot_line_with_sleeve(canopies_vector, 2 * canopies_stds, 'green')
        viz_utils.plot_line_with_sleeve(trunks_vector, 2 * trunks_stds, 'red')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, '%s.png' % plot_name))

    def clean_env(self):
        utils.kill_process('amcl')
        ros_utils.kill_master()

    def prologue(self):
        origin_map_image_path = self.data_sources['map_image_path']
        map_semantic_trunks = self.data_sources['map_semantic_trunks']
        localization_semantic_trunks = self.data_sources['localization_semantic_trunks']
        origin_localization_image_path = self.data_sources['localization_image_path']
        trajectory_waypoints = self.data_sources['trajectory_waypoints']
        bounding_box_expand_ratio = self.params['bounding_box_expand_ratio']
        mean_trunk_radius_for_map = self.params['mean_trunk_radius_for_map']
        mean_trunk_radius_for_localization = self.params['mean_trunk_radius_for_localization']
        std_trunk_radius_for_map = self.params['std_trunk_radius_for_map']
        std_trunk_radius_for_localization = self.params['std_trunk_radius_for_localization']

        # Generate canopies and trunk map images
        map_image = cv2.imread(origin_map_image_path)
        cv2.imwrite(os.path.join(self.experiment_dir, 'image_for_map.jpg'), map_image)
        canopies_map_image = maps_generation.generate_canopies_map(map_image)
        trunks_map_image = maps_generation.generate_trunks_map(map_image, map_semantic_trunks.values(),
                                                               mean_trunk_radius_for_map, std_trunk_radius_for_map,
                                                               np_random_state=self.np_random_state) # TODO: verify that std is not too small!!!
        upper_left, lower_right = cv_utils.get_bounding_box(canopies_map_image, map_semantic_trunks.values(), expand_ratio=bounding_box_expand_ratio)
        canopies_map_image = canopies_map_image[upper_left[1]:lower_right[1], upper_left[0]:lower_right[0]]
        trunks_map_image = trunks_map_image[upper_left[1]:lower_right[1], upper_left[0]:lower_right[0]]
        self.canopies_map_yaml_path, _ = ros_utils.save_image_to_map(canopies_map_image, resolution=self.params['map_resolution'], map_name='canopies_map', dir_name=self.experiment_dir)
        self.trunks_map_yaml_path, _ = ros_utils.save_image_to_map(trunks_map_image, resolution=self.params['map_resolution'], map_name='trunks_map', dir_name=self.experiment_dir)

        # Generate canopies and trunks localization images
        localization_image = cv2.imread(origin_localization_image_path)
        if origin_localization_image_path != origin_map_image_path:
            localization_image, affine_transform = cv_utils.warp_image(localization_image, localization_semantic_trunks.values(),
                                                                       map_semantic_trunks.values(), method='affine')
            localization_semantic_trunks_np = np.float32(localization_semantic_trunks.values()).reshape(-1, 1, 2)
            affine_transform = np.insert(affine_transform, [2], [0, 0, 1], axis=0) # TODO: double check if this is correct
            localization_semantic_trunks_np = cv2.perspectiveTransform(localization_semantic_trunks_np, affine_transform)
            localization_semantic_trunks = {key: (int(np.round(value[0])), int(np.round(value[1]))) for key, value
                                            in zip(map_semantic_trunks.keys(), localization_semantic_trunks_np[:, 0, :].tolist())}
            image_for_trajectory_path = os.path.join(self.experiment_dir, 'aligned_image_for_localization.jpg')
            cv2.imwrite(image_for_trajectory_path, localization_image)
        else:
            image_for_trajectory_path = origin_localization_image_path
        canopies_localization_image = maps_generation.generate_canopies_map(localization_image)
        trunks_localization_image = maps_generation.generate_trunks_map(localization_image, localization_semantic_trunks.values(),
                                                                        mean_trunk_radius_for_localization, std_trunk_radius_for_localization,
                                                                        np_random_state=self.np_random_state)
        canopies_localization_image = canopies_localization_image[upper_left[1]:lower_right[1], upper_left[0]:lower_right[0]]
        trunks_localization_image = trunks_localization_image[upper_left[1]:lower_right[1], upper_left[0]:lower_right[0]]
        self.canopies_localization_image_path = os.path.join(self.experiment_dir, 'canopies_localization.jpg')
        cv2.imwrite(self.canopies_localization_image_path, canopies_localization_image)
        self.trunks_localization_image_path = os.path.join(self.experiment_dir, 'trunks_localization.jpg')
        cv2.imwrite(self.trunks_localization_image_path, trunks_localization_image)
        self.canopies_localization_image_height = canopies_localization_image.shape[0]
        self.trunks_localization_image_height = trunks_localization_image.shape[0]
        viz_image = cv2.imread(image_for_trajectory_path)[upper_left[1]:lower_right[1], upper_left[0]:lower_right[0]]
        self.viz_image_path = os.path.join(self.experiment_dir, 'image_for_visualization.jpg')
        cv2.imwrite(self.viz_image_path, viz_image)

        # Get trajectory
        waypoints_coordinates = []
        for waypoint in trajectory_waypoints:
            if type(waypoint) is tuple:
                point1 = localization_semantic_trunks[waypoint[0]]
                point2 = localization_semantic_trunks[waypoint[1]]
                waypoints_coordinates.append(((point1[0] + point2[0]) / 2, (point1[1] + point2[1]) / 2))
            else:
                waypoints_coordinates.append(localization_semantic_trunks[waypoint])
        path_planning_experiment = PathPlanningExperiment(name='path_planning',
                                                          data_sources={'map_image_path': image_for_trajectory_path,
                                                                        'map_upper_left': upper_left, 'map_lower_right': lower_right,
                                                                        'waypoints': waypoints_coordinates},
                                                          working_dir=self.experiment_dir, metadata=self.metadata)
        path_planning_experiment.run(repetitions=1)
        self.results['trajectory'] = path_planning_experiment.results[1]['trajectory']
        trajectory = self.results['trajectory']
        freq = self.params['target_frequency']
        epsilon_t = 1e-3 # TODO: this is new, suspect here
        timestamps = np.linspace(start=epsilon_t, stop=epsilon_t + (1.0 / freq) * len(trajectory), num=len(trajectory))
        pose_time_tuples_list = [(x, y, t) for (x, y), t in zip(trajectory, timestamps)]
        self.trajectory_bag_path = os.path.join(self.experiment_dir, 'trajectory.bag')
        ros_utils.trajectory_to_bag(pose_time_tuples_list, self.trajectory_bag_path)

        # Generate scan offline
        self.canopies_scans_pickle_path = os.path.join(self.experiment_dir, 'canopies_scan.pkl')
        offline_synthetic_scan_generator.generate_scans_pickle(self.trajectory_bag_path, canopies_localization_image, self.params['min_angle'],
                                                               self.params['max_angle'], self.params['samples_num'], self.params['min_distance'],
                                                               self.params['max_distance'], self.params['localization_resolution'],
                                                               self.params['r_primary_search_samples'], self.params['r_secondary_search_step'],
                                                               output_pickle_path=self.canopies_scans_pickle_path)
        self.trunks_scans_pickle_path = os.path.join(self.experiment_dir, 'trunks_scan.pkl')
        offline_synthetic_scan_generator.generate_scans_pickle(self.trajectory_bag_path, trunks_localization_image, self.params['min_angle'],
                                                               self.params['max_angle'], self.params['samples_num'], self.params['min_distance'],
                                                               self.params['max_distance'], self.params['localization_resolution'],
                                                               self.params['r_primary_search_samples'], self.params['r_secondary_search_step'],
                                                               output_pickle_path=self.trunks_scans_pickle_path)

        # Calculate valid scans rate
        with open(self.canopies_scans_pickle_path) as f:
            canopies_scans = pickle.load(f)
        with open(self.trunks_scans_pickle_path) as f:
            trunks_scans = pickle.load(f)
        canopies_invalid_scans = np.sum([np.all(np.isnan(scan)) for scan in canopies_scans.values()])
        canopies_total_timestamps = len(canopies_scans.values())
        self.results['canopies_valid_scans_rate'] = float(canopies_total_timestamps - canopies_invalid_scans) / canopies_total_timestamps
        trunks_invalid_scans = np.sum([np.all(np.isnan(scan)) for scan in trunks_scans.values()])
        trunks_total_timestamps = len(trunks_scans.values())
        self.results['trunks_valid_scans_rate'] = float(trunks_total_timestamps - trunks_invalid_scans) / trunks_total_timestamps


    def epilogue(self):
        # Plot graphs
        canopies_errors, canopies_covariance_norms = self._get_errors_and_covariance_norms_dataframes(namespace='canopies')
        canopies_mean_errors = canopies_errors.mean(axis=1)
        canopies_std_errors = canopies_errors.std(axis=1)
        canopies_mean_covariance_norms = canopies_covariance_norms.mean(axis=1)
        canopies_std_covariance_norms = canopies_covariance_norms.std(axis=1)
        trunks_errors, trunks_covariance_norms = self._get_errors_and_covariance_norms_dataframes(namespace='trunks')
        trunks_mean_errors = trunks_errors.mean(axis=1)
        trunks_std_errors = trunks_errors.std(axis=1)
        trunks_mean_covariance_norms = trunks_covariance_norms.mean(axis=1)
        trunks_std_covariance_norms = trunks_covariance_norms.std(axis=1)
        self._plot_canopies_vs_trunks('error', canopies_mean_errors, trunks_mean_errors, canopies_std_errors,
                                      trunks_std_errors, self.experiment_dir)
        self._plot_canopies_vs_trunks('covariance_norm', canopies_mean_covariance_norms, trunks_mean_covariance_norms,
                                      canopies_std_covariance_norms, trunks_std_covariance_norms, self.experiment_dir)

    def task(self, **kwargs):

        launch_rviz = kwargs.get('launch_rviz', False)

        # Start ROS and RVIZ
        ros_utils.start_master()
        if launch_rviz:
            ros_utils.launch_rviz(os.path.join(config.root_dir_path, 'src/experiments_framework/framework/amcl.rviz'))

        # Launch localization stack for canopies and trunks
        self._launch_world_to_map_static_tf(namespace='canopies')
        self._launch_world_to_map_static_tf(namespace='trunks')
        self._launch_base_to_scan_static_tf(namespace='canopies')
        self._launch_base_to_scan_static_tf(namespace='trunks')
        self._launch_synthetic_scan_generator(self.canopies_localization_image_path, namespace='canopies', pickle_path=self.canopies_scans_pickle_path)
        self._launch_synthetic_scan_generator(self.trunks_localization_image_path, namespace='trunks', pickle_path=self.trunks_scans_pickle_path)
        self._launch_map_server(self.canopies_map_yaml_path, namespace='canopies')
        self._launch_map_server(self.trunks_map_yaml_path, namespace='trunks')
        self._launch_amcl(namespace='canopies')
        self._launch_amcl(namespace='trunks')
        self._initialize_global_localization(namespace='canopies')
        self._initialize_global_localization(namespace='trunks')
        self._launch_synthetic_odometry(namespace='canopies')
        self._launch_synthetic_odometry(namespace='trunks')

        if launch_rviz:
            self._launch_contours_visualization()

        # Start recording output bag
        output_bag_path = os.path.join(self.repetition_dir, '%s_output.bag' % self.name)
        ros_utils.start_recording_bag(output_bag_path, ['/ugv_pose', '/canopies/amcl_pose', '/canopies/particlecloud',
                                                        '/trunks/amcl_pose', '/trunks/particlecloud']) # TODO: exceptions are thrown!!!

        # Start input bag and wait
        _, bag_duration = ros_utils.play_bag(self.trajectory_bag_path)
        time.sleep(bag_duration)

        # Stop recording output bag
        ros_utils.stop_recording_bags()

        # Kill RVIZ
        if launch_rviz:
            ros_utils.kill_rviz()

        # Generate results dafaframes
        self._generate_results_dataframe(namespace='canopies', output_bag_path=output_bag_path, image_height=self.canopies_localization_image_height)
        self._generate_results_dataframe(namespace='trunks', output_bag_path=output_bag_path, image_height=self.trunks_localization_image_height)

        # Delete bag file
        os.remove(output_bag_path)