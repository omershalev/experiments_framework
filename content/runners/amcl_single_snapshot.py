import os
import json
import numpy as np
import pandas as pd

from framework import utils
from framework import ros_utils
from framework import config
from content.experiments.amcl_snapshots import AmclSnapshotsExperiment
from content.data_pointers.lavi_april_18.dji import trunks_detection_results_dir, selected_trunks_detection_experiments_and_repetitions

repetitions = 1
samples_num = 10

if __name__ == '__main__':
    execution_dir = utils.create_new_execution_folder('amcl_single_image')

    for trunks_detection_experiment_name, trunks_detection_repetition in selected_trunks_detection_experiments_and_repetitions:
        with open (os.path.join(trunks_detection_results_dir, trunks_detection_experiment_name, 'experiment_summary.json')) as f:
            trunks_detection_summary = json.load(f)
        image_path = trunks_detection_summary['data_sources']
        image_key = trunks_detection_summary['metadata']['image_key']
        semantic_trunks = trunks_detection_summary['results'][str(trunks_detection_repetition)]['semantic_trunks']
        for key in semantic_trunks.keys(): # TODO: formalize...
            if '1' in key or '2' in key or '10' in key or '9' in key or 'I' in key or 'H' in key: # Worked
            # if '1' in key or '10' in key or 'I' in key or 'H' in key: # Worked also!
                del semantic_trunks[key]
        experiment = AmclSnapshotsExperiment(name='amcl_single_snapshot_on_%s' % image_key,
                                             data_sources={'localization_image_path': image_path, 'map_image_path': image_path, 'semantic_trunks': semantic_trunks},
                                             params={'odometry_source': 'synthetic',
                                                     'odometry_noise_sigma': 0.03, # TODO: READ FROM CONFIG OR CHANGE!!!!!!!!
                                                     'bounding_box_expand_ratio': config.bounding_box_expand_ratio,
                                                     'gaussian_scale_factor': config.cost_map_gaussians_scale_factor,
                                                     'min_angle': config.synthetic_scan_min_angle,
                                                     'max_angle': config.synthetic_scan_max_angle,
                                                     'samples_num': config.synthetic_scan_samples_num,
                                                     'min_distance': config.synthetic_scan_min_distance,
                                                     'max_distance': config.synthetic_scan_max_distance,
                                                     'resolution': config.top_view_resolution,
                                                     'r_primary_search_samples': config.synthetic_scan_r_primary_search_samples,
                                                     'r_secondary_search_step': config.synthetic_scan_r_secondary_search_step},
                                             metadata=trunks_detection_summary['metadata'],
                                             working_dir=execution_dir)
        experiment.run(repetitions, launch_rviz=True)

        # Post processing
        joint_amcl_results_df = pd.DataFrame()
        for repetition_id in experiment.valid_repetitions:
            amcl_results_df = pd.read_csv(experiment.results[repetition_id]['amcl_results_path'], index_col=0)
            joint_amcl_results_df = pd.concat([joint_amcl_results_df, amcl_results_df], axis=1)


        error_samples_df = pd.DataFrame()
        covariance_norm_samples_df = pd.DataFrame()
        delta_t = (joint_amcl_results_df.index[-1] - joint_amcl_results_df.index[0]) / samples_num
        search_timestamps = [joint_amcl_results_df.index[joint_amcl_results_df.index > delta_t * i][0] for i in range(1, samples_num + 1)]
        for repetition_id in experiment.valid_repetitions:
            valid_repetition_indices = np.where(~joint_amcl_results_df['amcl_pose_error[%d]' % repetition_id].isnull())[0]
            this_repetition_errors = []
            this_repetition_covaraiance_norms = []
            for search_timestamp in search_timestamps:
                search_index = joint_amcl_results_df.index.get_loc(search_timestamp)
                def find_nearest(array, value):
                    # array = np.asarray(array)
                    idx = (np.abs(array - value)).argmin()
                    return array[idx]
                nearest_valid_index = find_nearest(valid_repetition_indices, search_index)
                this_repetition_errors.append(joint_amcl_results_df.iloc[nearest_valid_index]['amcl_pose_error[%d]' % repetition_id])
                this_repetition_covaraiance_norms.append(joint_amcl_results_df.iloc[nearest_valid_index]['amcl_covariance_norm[%d]' % repetition_id])
            error_samples_df = pd.concat([error_samples_df, pd.Series(this_repetition_errors)], axis=1)
            covariance_norm_samples_df = pd.concat([covariance_norm_samples_df, pd.Series(this_repetition_covaraiance_norms)], axis=1)
        print ('break')