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


class IcpSimulationExperiment(Experiment):

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

    def _launch_icp(self, namespace):
        ros_utils.launch(package='localization',
                         launch_file='scan_matcher.launch',
                         argv={'ns': namespace})

    def _generate_results_dataframe(self, namespace, output_bag_path, image_height):
        ground_truth_df = ros_utils.bag_to_dataframe(output_bag_path, topic='/ugv_pose', fields=['point.x', 'point.y'])
        ground_truth_df['point.x'] = ground_truth_df['point.x'].apply(lambda cell: cell * self.params['localization_resolution'])
        ground_truth_df['point.y'] = ground_truth_df['point.y'].apply(lambda cell: (image_height - cell) * self.params['localization_resolution'])
        ground_truth_df.columns = ['ground_truth_x[%d]' % self.repetition_id, 'ground_truth_y[%d]' % self.repetition_id]
        icp_pose_df = ros_utils.bag_to_dataframe(output_bag_path, topic='/%s/scanmatcher_pose' % namespace, fields=['x', 'y'])
        icp_pose_df.columns = ['icp_pose_x[%d]' % (self.repetition_id), 'icp_pose_y[%d]' % (self.repetition_id)]
        ground_truth_df = ground_truth_df[~ground_truth_df.index.duplicated(keep='first')]
        icp_pose_df = icp_pose_df[~icp_pose_df.index.duplicated(keep='first')]
        icp_results_df = pd.concat([ground_truth_df, icp_pose_df], axis=1)
        icp_results_df['ground_truth_x[%d]' % self.repetition_id] = icp_results_df['ground_truth_x[%d]' % self.repetition_id].interpolate()
        icp_results_df['ground_truth_y[%d]' % self.repetition_id] = icp_results_df['ground_truth_y[%d]' % self.repetition_id].interpolate()
        icp_results_path = os.path.join(self.repetition_dir, '%s_icp_results.csv' % namespace)
        icp_results_df.to_csv(icp_results_path)
        self.results[self.repetition_id]['%s_icp_results_path' % namespace] = icp_results_path

    def clean_env(self):
        utils.kill_process('laser_scan_matcher') # TODO: verify!!!
        ros_utils.kill_master()

    def prologue(self):
        localization_semantic_trunks = self.data_sources['localization_semantic_trunks']
        origin_localization_image_path = self.data_sources['localization_image_path']
        trajectory_waypoints = self.data_sources['trajectory_waypoints']
        bounding_box_expand_ratio = self.params['bounding_box_expand_ratio']
        mean_trunk_radius_for_localization = self.params['mean_trunk_radius_for_localization']
        std_trunk_radius_for_localization = self.params['std_trunk_radius_for_localization']
        localization_external_trunks = self.data_sources['localization_external_trunks']

        # Generate canopies and trunks localization images
        localization_image = cv2.imread(origin_localization_image_path)

        image_for_trajectory_path = origin_localization_image_path

        canopies_localization_image = maps_generation.generate_canopies_map(localization_image)

        upper_left, lower_right = cv_utils.get_bounding_box(canopies_localization_image,
                                                            localization_semantic_trunks.values(),
                                                            expand_ratio=bounding_box_expand_ratio)

        canopies_localization_image = canopies_localization_image[upper_left[1]:lower_right[1],
                                      upper_left[0]:lower_right[0]]
        self.canopies_localization_image_path = os.path.join(self.experiment_dir, 'canopies_localization.jpg')
        cv2.imwrite(self.canopies_localization_image_path, canopies_localization_image)
        self.canopies_localization_image_height = canopies_localization_image.shape[0]
        trunks_localization_image = maps_generation.generate_trunks_map(localization_image, localization_semantic_trunks.values() + localization_external_trunks,
                                                                        mean_trunk_radius_for_localization, std_trunk_radius_for_localization,
                                                                        np_random_state=self.np_random_state)
        trunks_localization_image = trunks_localization_image[upper_left[1]:lower_right[1], upper_left[0]:lower_right[0]]
        self.trunks_localization_image_path = os.path.join(self.experiment_dir, 'trunks_localization.jpg')
        cv2.imwrite(self.trunks_localization_image_path, trunks_localization_image)
        self.trunks_localization_image_height = trunks_localization_image.shape[0]

        # Get trajectory
        waypoints_coordinates = []
        for waypoint in trajectory_waypoints:
            if type(waypoint) is tuple and len(waypoint) == 2:
                point1 = localization_semantic_trunks[waypoint[0]]
                point2 = localization_semantic_trunks[waypoint[1]]
                waypoints_coordinates.append(((point1[0] + point2[0]) / 2, (point1[1] + point2[1]) / 2))
            elif type(waypoint) is tuple and len(waypoint) == 6:
                point1 = np.array(localization_semantic_trunks[waypoint[0]]) + np.array([waypoint[2], waypoint[3]])
                point2 = np.array(localization_semantic_trunks[waypoint[1]]) + np.array([waypoint[4], waypoint[5]])
                waypoints_coordinates.append(((point1[0] + point2[0]) / 2, (point1[1] + point2[1]) / 2))
            else:
                waypoints_coordinates.append(localization_semantic_trunks[waypoint])
        path_planning_experiment = PathPlanningExperiment(name='path_planning',
                                                          data_sources={'map_image_path': image_for_trajectory_path,
                                                                        'map_upper_left': upper_left,
                                                                        'map_lower_right': lower_right,
                                                                        'waypoints': waypoints_coordinates},
                                                          working_dir=self.experiment_dir, metadata=self.metadata)
        path_planning_experiment.run(repetitions=1)
        self.results['trajectory'] = path_planning_experiment.results[1]['trajectory']
        trajectory = self.results['trajectory']
        freq = self.params['target_frequency']
        epsilon_t = 1e-3
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

    def task(self, **kwargs):

        # Start ROS and RVIZ
        ros_utils.start_master()

        # Launch localization stack for canopies and trunks
        self._launch_base_to_scan_static_tf(namespace='canopies')
        self._launch_base_to_scan_static_tf(namespace='trunks')
        self._launch_synthetic_scan_generator(self.canopies_localization_image_path, namespace='canopies', pickle_path=self.canopies_scans_pickle_path)
        self._launch_synthetic_scan_generator(self.trunks_localization_image_path, namespace='trunks', pickle_path=self.trunks_scans_pickle_path)
        self._launch_icp(namespace='canopies')
        self._launch_icp(namespace='trunks')

        # Start recording output bag
        output_bag_path = os.path.join(self.repetition_dir, '%s_output.bag' % self.name)
        ros_utils.start_recording_bag(output_bag_path, ['/ugv_pose', '/canopies/scanmatcher_pose', '/trunks/scanmatcher_pose'])

        # Start input bag and wait
        _, bag_duration = ros_utils.play_bag(self.trajectory_bag_path)
        time.sleep(bag_duration)

        # Stop recording output bag
        ros_utils.stop_recording_bags()

        # Generate results dafaframes
        self._generate_results_dataframe(namespace='canopies', output_bag_path=output_bag_path, image_height=self.canopies_localization_image_height)
        self._generate_results_dataframe(namespace='trunks', output_bag_path=output_bag_path, image_height=self.trunks_localization_image_height)

        # Delete bag file
        os.remove(output_bag_path)