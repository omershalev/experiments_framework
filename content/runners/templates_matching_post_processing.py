import os
import pandas as pd
import matplotlib.pyplot as plt

from framework import utils
from framework.config import base_results_path

#################################################################################################
#                                             CONFIG                                            #
#################################################################################################
template_matching_experiments_path = os.path.join(base_results_path, 'template_matching', 'template_matching_colored')
#################################################################################################

if __name__ == '__main__':
    execution_dir = utils.create_new_execution_folder('template_matching_post_processing')
    for template_matching_experiment in os.listdir(template_matching_experiments_path):
        errors_df = pd.read_csv(os.path.join(template_matching_experiments_path, template_matching_experiment, 'errors.csv'), index_col=0)
        errors_df = errors_df.dropna(how='all')
        plt.figure()
        for method in errors_df.columns:
            plt.plot(range(errors_df[method].shape[0]), errors_df[method])
        plt.ylim([0, 20])
        plt.savefig(os.path.join(execution_dir, '%s.jpg' % template_matching_experiment))