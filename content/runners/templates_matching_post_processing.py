import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import OrderedDict

from framework import utils
from framework import viz_utils
from framework.config import base_results_path

#################################################################################################
#                                             CONFIG                                            #
#################################################################################################
# template_matching_experiment_path = os.path.join(base_results_path, 'template_matching')
template_matching_experiment_path = r'/home/omer/temp/20190308-132822_template_matching/20190308-132822_template_matching_15-08-1_to_19-04-1v'
#################################################################################################






if __name__ == '__main__':
    execution_dir = utils.create_new_execution_folder('template_matching_post_processing')
    errors_df = pd.read_csv(os.path.join(template_matching_experiment_path, 'errors.csv'), index_col=0)
    errors_df = errors_df.dropna(how='all')
    for method in ['TM_CCOEFF', 'TM_CCORR']:
        plt.plot(range(errors_df[method].shape[0]), errors_df[method])
    plt.show()
    print ('end')
    # results_list = []
    # fig = plt.figure(figsize=(8.5, 11))
