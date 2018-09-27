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

class AmclSnapshotsExperiment(Experiment):

    def clean_env(self):
        utils.kill_process('amcl')
        ros_utils.kill_master()


    def task(self, **kwargs):

        launch_rviz = kwargs.get('launch_rviz', False)

        origin_map_image_path = self.data_sources['map_image_path']
        semantic_trunks = self.data_sources['semantic_trunks'] # TODO: this will become: map semantic_trunks (localization_semantic_trunks will be added and used for alignment)
        gaussian_scale_factor = self.params['gaussian_scale_factor']
        bounding_box_expand_ratio = self.params['bounding_box_expand_ratio']
        origin_localization_image_path = self.data_sources['localization_image_path']

        # Generate map image
        map_image = cv2.imread(origin_map_image_path)
        map_image = maps_generation.generate_canopies_map(map_image)
        upper_left, lower_right = cv_utils.get_bounding_box(map_image, semantic_trunks.values(), expand_ratio=bounding_box_expand_ratio)
        map_image = map_image[upper_left[1]:lower_right[1], upper_left[0]:lower_right[0]]
        map_yaml_path, _ = ros_utils.save_image_to_map(map_image, resolution=self.params['resolution'], map_name='map', dir_name=self.repetition_dir)

        # Generate localization image
        localization_image = cv2.imread(origin_localization_image_path)
        # TODO: with two images, localization_image will have to be warped to map_image here (and only then cropped, according to the following line)
        localization_image = maps_generation.generate_canopies_map(localization_image)
        localization_image = localization_image[upper_left[1]:lower_right[1], upper_left[0]:lower_right[0]]
        localization_image_path = os.path.join(self.repetition_dir, 'localization.jpg')
        cv2.imwrite(localization_image_path, localization_image)

        ##### WIP #####
        trunk_radius = 15 # TODO: change!
        wp1 = tuple(np.array(semantic_trunks['7/A']) + np.array([trunk_radius + 70, 0]))
        wp2 = tuple(np.array(semantic_trunks['7/G']) + np.array([trunk_radius + 115, 0]))
        # wp3 = tuple(np.array(semantic_trunks['7/A']) + np.array([trunk_radius + 70, 0]))
        # wp4 = tuple(np.array(semantic_trunks['6/A']) + np.array([trunk_radius + 70, 0]))
        # wp5 = tuple(np.array(semantic_trunks['6/G']) + np.array([trunk_radius + 115, 0]))
        # wp6 = tuple(np.array(semantic_trunks['6/A']) + np.array([trunk_radius + 70, 0]))
        # waypoints = [wp1, wp2, wp3, wp4, wp5, wp6] # TODO: rearrange
        waypoints = [wp1, wp2] # TODO: rearrange
        optimized_sigma = 90.39 # TODO: rearrange
        freq = 30 # TODO: rearrange and consider what frequency to use!!!
        experiment = PathPlanningExperiment(name='path_planning',
                                            data_sources={'map_image_path': origin_map_image_path, 'trunk_points_list': semantic_trunks.values(),
                                                          'map_upper_left': upper_left, 'map_lower_right': lower_right, 'waypoints': waypoints},
                                            params={'trunk_radius': trunk_radius, 'gaussian_scale_factor': gaussian_scale_factor,
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

        # Launch base_link to contours_scan_link static TF
        ros_utils.launch(package='localization',
                         launch_file='static_identity_tf.launch',
                         argv={'frame_id': 'base_link', 'child_frame_id': 'contours_scan_link'})

        # Launch synthetic scan generator
        ros_utils.launch(package='localization',
                         launch_file='synthetic_scan_generator.launch',
                         argv={'virtual_ugv_mode': True,
                               'localization_image_path': localization_image_path,
                               'min_angle': self.params['min_angle'],
                               'max_angle': self.params['max_angle'],
                               'samples_num': self.params['samples_num'],
                               'min_distance': self.params['min_distance'],
                               'max_distance': self.params['max_distance'],
                               'resolution': self.params['resolution'],
                               'r_primary_search_samples': self.params['r_primary_search_samples'],
                               'r_secondary_search_step': self.params['r_secondary_search_step']})

        # Launch map server
        ros_utils.launch(package='localization',
                         launch_file='map.launch',
                         argv={'map_yaml_path': map_yaml_path})

        # Wait for map server to load
        ros_utils.wait_for_rosout_message(node_name='map_server',
                                          desired_message=r'Read a \d+ X \d+ map @ \d\.?\d+? m/cell',
                                          is_regex=True)

        # Launch AMCL
        ros_utils.launch(package='localization', launch_file='amcl.launch')

        # Wait for AMCL to load
        ros_utils.wait_for_rosout_message(node_name='amcl', desired_message='Done initializing likelihood field model.')

        # Launch odometry
        odometry_source = self.params['odometry_source']
        if odometry_source == 'icp':
            ros_utils.launch(package='localization', launch_file='icp.launch')
        elif odometry_source == 'synthetic':
            ros_utils.launch(package='localization', launch_file='synthetic_odometry.launch',
                             argv={'resolution': self.params['resolution'],
                                   'noise_sigma': self.params['odometry_noise_sigma']})
        else:
            raise Exception('Unknown odometry source %s' % odometry_source)

        # Start recording output bag
        output_bag_path = os.path.join(self.repetition_dir, '%s_output.bag' % self.name)
        self.results[self.repetition_id]['output_bag_path'] = output_bag_path
        ros_utils.start_recording_bag(output_bag_path, ['/amcl_pose', '/particlecloud', '/scanmatcher_pose', '/ugv_pose'])

        # Start input bag and wait
        _, bag_duration = ros_utils.play_bag(trajectory_bag_path)
        time.sleep(bag_duration)

        # Stop recording output bag
        ros_utils.stop_recording_bags()

        # Kill RVIZ
        if launch_rviz:
            ros_utils.kill_rviz()

        # Generate results dafaframe
        ground_truth_df = ros_utils.bag_to_dataframe(output_bag_path, topic='/ugv_pose', fields=['x', 'y'])
        ground_truth_df['x'] = ground_truth_df['x'].apply(lambda cell: cell * config.top_view_resolution)
        ground_truth_df['y'] = ground_truth_df['y'].apply(lambda cell: (localization_image.shape[0] - cell) * config.top_view_resolution)  # TODO: is this the height of the localization image??
        ground_truth_df.columns = ['ground_truth_x[%d]' % self.repetition_id, 'ground_truth_y[%d]' % self.repetition_id]
        amcl_pose_df = ros_utils.bag_to_dataframe(output_bag_path, topic='/amcl_pose', fields=['pose.pose.position.x', 'pose.pose.position.y'])
        amcl_pose_df.columns = ['amcl_pose_x[%d]' % self.repetition_id, 'amcl_pose_y[%d]' % self.repetition_id]
        def covariance_norm(covariance_mat): # TODO: think about this with Amir
            return np.linalg.norm(np.array(covariance_mat).reshape(6, 6))
        amcl_covariance_df = ros_utils.bag_to_dataframe(output_bag_path, topic='/amcl_pose', fields=['pose.covariance'], aggregation=covariance_norm)
        amcl_covariance_df.columns = ['amcl_covariance_norm[%d]' % self.repetition_id]
        amcl_results_df = pd.concat([ground_truth_df, amcl_pose_df, amcl_covariance_df], axis=1)
        amcl_results_df['ground_truth_x[%d]' % self.repetition_id] = amcl_results_df['ground_truth_x[%d]' % self.repetition_id].interpolate()  # TODO: try method='time'
        amcl_results_df['ground_truth_y[%d]' % self.repetition_id] = amcl_results_df['ground_truth_y[%d]' % self.repetition_id].interpolate()  # TODO: try method='time'
        error = np.sqrt((amcl_results_df['ground_truth_x[%d]' % self.repetition_id] - amcl_results_df['amcl_pose_x[%d]' % self.repetition_id]) ** 2 + \
                        (amcl_results_df['ground_truth_y[%d]' % self.repetition_id] - amcl_results_df['amcl_pose_y[%d]' % self.repetition_id]) ** 2)
        amcl_results_df['amcl_pose_error[%d]' % self.repetition_id] = error
        amcl_results_path = os.path.join(self.repetition_dir, 'amcl_results.csv')
        amcl_results_df.to_csv(amcl_results_path)
        self.results[self.repetition_id]['amcl_results_path'] = amcl_results_path