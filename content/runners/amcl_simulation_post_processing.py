import os
import json
import pandas as pd
import numpy as np
from collections import OrderedDict

from framework import utils
from framework.config import base_results_path

#################################################################################################
#                                             CONFIG                                            #
#################################################################################################
original_execution_dirname = os.path.join('20190203-004518_amcl_baseline_almost_amazing', '15-08-1')
parameters_to_compare = ['min_amcl_particles', 'max_distance'] # if None, don't compare
error_band_width = 2
confidence_band_width = 200 # TODO: rethink!
#################################################################################################


def calculate_measures(df):
    first_time_in_band_values = []
    in_band_percentage_values = []
    for repetition_header in df.columns:
        repetition_vector = df[repetition_header]
        in_band = repetition_vector[repetition_vector < error_band_width]
        first_time_in_band_values.append(in_band.index[0] if len(in_band) > 0 else np.nan)
        in_band_percentage_values.append(1.0 * len(in_band) / len(repetition_vector))
    if np.isnan(first_time_in_band_values).sum() > len(df.columns) / 2.0: # TODO: elaborate about this logic in the thesis
        average_first_time_in_band = np.nan
    else:
        average_first_time_in_band = np.nanmean(first_time_in_band_values)
    average_in_band_percentage = np.mean(in_band_percentage_values)
    df_stds = df.std(axis=1)
    average_std = df_stds.mean()
    return average_first_time_in_band, average_in_band_percentage, average_std


if __name__ == '__main__':
    execution_dir = utils.create_new_execution_folder('amcl_simulation_post_processing')
    results_list = []
    for experiment_name in os.listdir(os.path.join(base_results_path, 'amcl', original_execution_dirname)):
        with open(os.path.join(base_results_path, 'amcl', original_execution_dirname, experiment_name, 'experiment_summary.json')) as f:
            experiment_summary = json.load(f)
        trajectory_name = experiment_summary['metadata']['trajectory_name']
        canopies_errors = pd.read_csv(experiment_summary['results']['canopies_errors_path'], index_col=0)
        canopies_covariance_norms = pd.read_csv(experiment_summary['results']['canopies_covariance_norms_path'], index_col=0)
        canopies_first_time_in_error_band, canopies_in_error_band_percentage, canopies_errors_std = calculate_measures(canopies_errors)
        canopies_first_time_in_confidence_band, canopies_in_confidence_band_percentage, canopies_covariance_norms_std = calculate_measures(canopies_covariance_norms)
        trunks_errors = pd.read_csv(experiment_summary['results']['trunks_errors_path'], index_col=0)
        trunks_covariance_norms = pd.read_csv(experiment_summary['results']['trunks_covariance_norms_path'], index_col=0)
        trunks_first_time_in_error_band, trunks_in_error_band_percentage, trunks_errors_std = calculate_measures(trunks_errors)
        trunks_first_time_in_confidence_band, trunks_in_confidence_band_percentage, trunks_covariance_norms_std = calculate_measures(trunks_covariance_norms)
        index = trajectory_name
        if parameters_to_compare is not None:
            index += '___' + '__'.join(['%s=%s' % (parameter, experiment_summary['params'][parameter]) for parameter in parameters_to_compare])
        entry = pd.DataFrame(data=OrderedDict([
            ('canopies/valid_scans_rate', experiment_summary['results']['canopies_valid_scans_rate']),
            ('trunks/valid_scans_rate', experiment_summary['results']['trunks_valid_scans_rate']),
            ('canopies/average_in_error_band_percentage', canopies_in_error_band_percentage),
            ('trunks/average_in_error_band_percentage', trunks_in_error_band_percentage),
            ('canopies/average_first_time_in_error_band', canopies_first_time_in_error_band),
            ('trunks/average_first_time_in_error_band', trunks_first_time_in_error_band),
            ('canopies/average_in_confidence_band_percentage', canopies_in_confidence_band_percentage),
            ('trunks/average_in_confidence_band_percentage', trunks_in_confidence_band_percentage),
            ('canopies/average_first_time_in_confidence_band', canopies_first_time_in_confidence_band),
            ('trunks/average_first_time_in_confidence_band', trunks_first_time_in_confidence_band),
            ('canopies/average_errors_std', canopies_errors_std),
            ('trunks/average_errors_std', trunks_errors_std),
            ('canopies/average_covariance_norms_std', canopies_covariance_norms_std),
            ('trunks/average_covariance_norms_std', trunks_covariance_norms_std)]),
            index=[index])
        if parameters_to_compare is not None:
            for parameter in parameters_to_compare:
                entry.loc[index, parameter] = experiment_summary['params'][parameter]
        results_list.append(entry)
    results_df = pd.concat(results_list)
    results_df.to_csv(os.path.join(execution_dir, 'amcl_aggregation.csv'))