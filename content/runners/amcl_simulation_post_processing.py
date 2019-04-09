import os
import json
from collections import OrderedDict
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from framework import utils
from framework import viz_utils
from framework.config import base_results_path

#################################################################################################
#                                             CONFIG                                            #
#################################################################################################
# original_execution_dirname = os.path.join('apr_basic', '15-08-1_15-53-1')
# original_execution_dirname = os.path.join('apr_basic', '15-08-1_16-55-1')
# original_execution_dirname = os.path.join('apr_basic', '15-08-1_19-04-1')
# original_execution_dirname = os.path.join('apr_basic', '15-53-1_16-55-1')
# original_execution_dirname = os.path.join('apr_basic', '15-53-1_19-04-1')
# original_execution_dirname = os.path.join('apr_basic', '16-55-1_19-04-1')
# original_execution_dirname = os.path.join('apr_noises_selected_instances', 'odometry_sigma_x=0.010000')
original_execution_dirname = os.path.join('apr_noises_selected_instances', 'scan_noise=0.100000')
parameters_to_compare = None # ['min_amcl_particles', 'max_distance'] # if None, don't compare
error_band_width = 1
confidence_band_width = 1
plot = False
#################################################################################################


def calculate_measures(df):
    first_time_in_band_values = []
    in_band_percentage_values = []
    for repetition_header in df.columns:
        repetition_vector = df[repetition_header]
        in_band = repetition_vector[repetition_vector < error_band_width]
        first_time_in_band_values.append(in_band.index[0] if len(in_band) > 0 else np.nan)
        in_band_percentage_values.append(1.0 * len(in_band) / len(repetition_vector))
    average_first_time_in_band = np.nanmean(first_time_in_band_values)
    convergence_to_band_ratio = 1.0 * (~np.isnan(first_time_in_band_values)).sum() / len(df.columns)
    average_in_band_ratio = np.mean(in_band_percentage_values)
    df_stds = df.std(axis=1)
    average_std = df_stds.mean()
    average_mean = df.mean().mean()
    return convergence_to_band_ratio, average_first_time_in_band, average_in_band_ratio, average_std, average_mean


def plot_canopies_vs_trunks(canopies_vector, trunks_vector, canopies_stds, trunks_stds, x_label=None, y_label=None):
    viz_utils.plot_line_with_sleeve(canopies_vector, 2 * canopies_stds, 'green')
    viz_utils.plot_line_with_sleeve(trunks_vector, 2 * trunks_stds, 'sienna')
    plt.autoscale(enable=True, axis='x', tight=True)
    plt.ylim(bottom=0)
    if x_label is not None:
        plt.xlabel(x_label)
    if y_label is not None:
        plt.ylabel(y_label)


if __name__ == '__main__':
    execution_dir = utils.create_new_execution_folder('amcl_simulation_post_processing_%s' % os.path.basename(original_execution_dirname))
    results_list = []
    if plot:
        fig = plt.figure(figsize=(8.5, 11))
        ax_errors = None
        ax_covariance_norms = None
    full_experiment_names = filter(lambda name: name.find('png') == -1, os.listdir(os.path.join(base_results_path, 'amcl', original_execution_dirname)))
    sorted_full_experiment_names = []
    for trajectory_name in ['narrow_row', 'wide_row', 's_patrol', 'u_turns', 'tasks_and_interrupts']:
        for full_experiment_name in full_experiment_names:
            if full_experiment_name.find(trajectory_name) != -1:
                sorted_full_experiment_names.append(full_experiment_name)
    for experiment_idx, experiment_name in enumerate(sorted_full_experiment_names):
        with open(os.path.join(base_results_path, 'amcl', original_execution_dirname, experiment_name, 'experiment_summary.json')) as f:
            experiment_summary = json.load(f)
        trajectory_name = experiment_summary['metadata']['trajectory_name']
        canopies_errors = pd.read_csv(os.path.join(base_results_path, 'amcl', original_execution_dirname, experiment_name, 'canopies_errors.csv'), index_col=0)
        canopies_covariance_norms = pd.read_csv(os.path.join(base_results_path, 'amcl', original_execution_dirname, experiment_name, 'canopies_covariance_norms.csv'), index_col=0)
        canopies_convergence_to_error_band_ratio, canopies_first_time_in_error_band, canopies_in_error_band_ratio, canopies_errors_std, _ = calculate_measures(canopies_errors)
        canopies_convergence_to_condifence_band_ratio, canopies_first_time_in_confidence_band, canopies_in_confidence_band_percentage, canopies_covariance_norms_std, canopies_average_mean_covariance_norm = calculate_measures(canopies_covariance_norms)
        trunks_errors = pd.read_csv(os.path.join(base_results_path, 'amcl', original_execution_dirname, experiment_name, 'trunks_errors.csv'), index_col=0)
        trunks_covariance_norms = pd.read_csv(os.path.join(base_results_path, 'amcl', original_execution_dirname, experiment_name, 'trunks_covariance_norms.csv'), index_col=0)
        trunks_convergence_to_error_band_ratio, trunks_first_time_in_error_band, trunks_in_error_band_percentage, trunks_errors_std, _ = calculate_measures(trunks_errors)
        trunks_convergence_to_confidence_band_ratio, trunks_first_time_in_confidence_band, trunks_in_confidence_band_percentage, trunks_covariance_norms_std, trunks_average_mean_covariance_norm = calculate_measures(trunks_covariance_norms)
        if plot:
            x_label = 't [sec]' if experiment_idx == 4 else None
            if ax_errors is None:
                ax_errors = fig.add_subplot(5, 2, 2 * experiment_idx + 1)
            else:
                fig.add_subplot(5, 2, 2 * experiment_idx + 1, sharey=ax_errors)
            plot_canopies_vs_trunks(canopies_errors.mean(axis=1), trunks_errors.mean(axis=1),
                                    canopies_errors.std(axis=1), trunks_errors.std(axis=1),
                                    x_label=x_label, y_label='avg. pose error [m]')
            if ax_covariance_norms is None:
                ax_covariance_norms = fig.add_subplot(5, 2, 2 * experiment_idx + 2)
            else:
                ax_covariance_norms = fig.add_subplot(5, 2, 2 * experiment_idx + 2, sharey=ax_covariance_norms)
            plot_canopies_vs_trunks(canopies_covariance_norms.mean(axis=1), trunks_covariance_norms.mean(axis=1),
                                    canopies_covariance_norms.std(axis=1), trunks_covariance_norms.std(axis=1),
                                    x_label=x_label, y_label=r'avg. cov. norm [m$^2$]')
        df_index = experiment_name
        if parameters_to_compare is not None:
            df_index += '___' + '__'.join(['%s=%s' % (parameter, experiment_summary['params'][parameter]) for parameter in parameters_to_compare])
        entry = pd.DataFrame(data=OrderedDict([
            ('canopies/valid_scans_rate', experiment_summary['results']['canopies_valid_scans_rate']),
            ('trunks/valid_scans_rate', experiment_summary['results']['trunks_valid_scans_rate']),
            ('canopies/convergence_to_error_band_ratio', canopies_convergence_to_error_band_ratio),
            ('trunks/convergence_to_error_band_ratio', trunks_convergence_to_error_band_ratio),
            ('canopies/average_in_error_band_ratio', canopies_in_error_band_ratio),
            ('trunks/average_in_error_band_ratio', trunks_in_error_band_percentage),
            ('canopies/average_first_time_in_error_band', canopies_first_time_in_error_band),
            ('trunks/average_first_time_in_error_band', trunks_first_time_in_error_band),
            ('canopies/average_in_confidence_band_ratio', canopies_in_confidence_band_percentage),
            ('trunks/average_in_confidence_band_ratio', trunks_in_confidence_band_percentage),
            ('canopies/average_first_time_in_confidence_band', canopies_first_time_in_confidence_band),
            ('trunks/average_first_time_in_confidence_band', trunks_first_time_in_confidence_band),
            ('canopies/average_errors_std', canopies_errors_std),
            ('trunks/average_errors_std', trunks_errors_std),
            ('canopies/average_covariance_norms_std', canopies_covariance_norms_std),
            ('trunks/average_covariance_norms_std', trunks_covariance_norms_std),
            ('canopies/average_mean_covariance_norm', canopies_average_mean_covariance_norm),
            ('trunks/average_mean_covariance_norm', trunks_average_mean_covariance_norm)]),
            index=[df_index])
        if parameters_to_compare is not None:
            for parameter in parameters_to_compare:
                entry.loc[df_index, parameter] = experiment_summary['params'][parameter]
        results_list.append(entry)
    if plot:
        fig.legend([Line2D([0], [0], color='green', linewidth=2), Line2D([0], [0], color='sienna', linewidth=2)],
                   ['canopies', 'trunks'], loc='lower left', ncol=1)
        plt.tight_layout()
        plt.savefig(os.path.join(execution_dir, 'graphs.jpg'))
    results_df = pd.concat(results_list)
    results_df.to_csv(os.path.join(execution_dir, 'amcl_aggregation.csv'))