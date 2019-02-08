import os
import json
import pandas as pd
from collections import OrderedDict

from framework import utils
from framework.config import base_results_path

#################################################################################################
#                                             CONFIG                                            #
#################################################################################################
original_execution_dir = os.path.join('20190203-004518_amcl_baseline_almost_amazing', '15-08-1')
error_band_width = 2
confidence_band_width = 200 # TODO: rethink!
#################################################################################################


if __name__ == '__main__':
    execution_dir = utils.create_new_execution_folder('amcl_simulation_post_processing')
    results_list = []
    for experiment_name in os.listdir(os.path.join(base_results_path, 'amcl', original_execution_dir)):
        with open(os.path.join(base_results_path, 'amcl', original_execution_dir, experiment_name, 'experiment_summary.json')) as f:
            experiment_summary = json.load(f)
        canopies_errors = pd.read_csv(experiment_summary['results']['canopies_errors_path'], index_col=0)
        canopies_covariance_norms = pd.read_csv(experiment_summary['results']['canopies_covariance_norms_path'], index_col=0)
        canopies_mean_errors = canopies_errors.mean(axis=1)
        canopies_mean_errors_band = canopies_mean_errors[canopies_mean_errors < error_band_width]
        canopies_mean_covariance_norms = canopies_covariance_norms.mean(axis=1)
        canopies_mean_covariance_norms_band = canopies_mean_covariance_norms[canopies_mean_covariance_norms < confidence_band_width]
        canopies_std_errors = canopies_errors.std(axis=1)
        canopies_std_covariance_norms = canopies_covariance_norms.std(axis=1)
        trunks_errors = pd.read_csv(experiment_summary['results']['trunks_errors_path'], index_col=0)
        trunks_covariance_norms = pd.read_csv(experiment_summary['results']['trunks_covariance_norms_path'], index_col=0)
        trunks_mean_errors = trunks_errors.mean(axis=1)
        trunks_mean_errors_band = trunks_mean_errors[trunks_mean_errors < error_band_width]
        trunks_mean_covariance_norms = trunks_covariance_norms.mean(axis=1)
        trunks_mean_covariance_norms_band = trunks_mean_covariance_norms[trunks_mean_covariance_norms < confidence_band_width]
        trunks_std_errors = trunks_errors.std(axis=1)
        trunks_std_covariance_norms = trunks_covariance_norms.std(axis=1)
        results_list.append(pd.DataFrame(data=OrderedDict([
                                    ('canopies/time_in_error_band', len(canopies_mean_errors_band)),
                                    ('canopies/first_time_in_error_band', canopies_mean_errors_band.index[0] if len(canopies_mean_errors_band) > 0 else pd.np.nan),
                                    ('canopies/mean_value_in_error_band', canopies_mean_errors_band.mean() if len(canopies_mean_errors_band) > 0 else pd.np.nan),
                                    ('canopies/time_in_confidence_band', len(canopies_mean_covariance_norms_band)),
                                    ('canopies/first_time_in_confidence_band', canopies_mean_covariance_norms_band.index[0] if len(canopies_mean_covariance_norms_band) > 0 else pd.np.nan),
                                    ('canopies/mean_value_in_confidence_band', canopies_mean_covariance_norms_band.mean() if len(canopies_mean_covariance_norms_band) > 0 else pd.np.nan),
                                    ('canopies/average_errors_std', canopies_std_errors.mean()),
                                    ('canopies/average_covariance_norms_std', canopies_std_covariance_norms.mean()),
                                    ('trunks/time_in_error_band', len(trunks_mean_errors_band)),
                                    ('trunks/first_time_in_error_band', trunks_mean_errors_band.index[0] if len(trunks_mean_errors_band) > 0 else pd.np.nan),
                                    ('trunks/mean_value_in_error_band', trunks_mean_errors_band.mean() if len(trunks_mean_errors_band) > 0 else pd.np.nan),
                                    ('trunks/time_in_confidence_band', len(trunks_mean_covariance_norms_band)),
                                    ('trunks/first_time_in_confidence_band', trunks_mean_covariance_norms_band.index[0] if len(trunks_mean_covariance_norms_band) > 0 else pd.np.nan),
                                    ('trunks/mean_value_in_confidence_band', trunks_mean_covariance_norms_band.mean() if len(trunks_mean_covariance_norms_band) > 0 else pd.np.nan),
                                    ('trunks/average_errors_std', trunks_std_errors.mean()),
                                    ('trunks/average_covariance_norms_std', trunks_std_covariance_norms.mean())]),
            index=[experiment_summary['metadata']['trajectory_name']]))
    results_df = pd.concat(results_list)
    print ('end')