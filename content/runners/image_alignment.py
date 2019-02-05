import os
import json
import pandas as pd
from itertools import combinations

from framework import utils
from content.experiments.image_alignment import ImageAlignmentExperiment


#################################################################################################
#                                             CONFIG                                            #
#################################################################################################
setup = 'apr' # apr / nov1
#################################################################################################

if setup == 'apr':
    from content.data_pointers.lavi_april_18.dji import trunks_detection_results_dir as td_results_dir
    from content.data_pointers.lavi_april_18.dji import selected_trunks_detection_experiments as selected_td_experiments
    from content.data_pointers.lavi_april_18.dji import snapshots_80_meters_markers_locations_json_path as markers_json_path
elif setup == 'nov1':
    from content.data_pointers.lavi_november_18.dji import trunks_detection_results_dir as td_results_dir
    from content.data_pointers.lavi_november_18.dji import selected_trunks_detection_experiments as selected_td_experiments
    from content.data_pointers.lavi_november_18.dji import snapshots_80_meters_markers_locations_json_path as markers_json_path
else:
    raise NotImplementedError


if __name__ == '__main__':
    execution_dir = utils.create_new_execution_folder('image_alignment_%s' % setup)
    results_df = pd.DataFrame(index=map(lambda c: '%s__%s' % (c[0], c[1]), combinations(selected_td_experiments, 2)),
                              columns=['by_trunks/mse', 'by_orb/mse', 'by_markers/mse', 'by_trunks/ssim', 'by_orb/ssim', 'by_markers/ssim'])
    with open(markers_json_path) as f:
        all_markers_locations = json.load(f)
    for experiment1, experiment2 in combinations(selected_td_experiments, r=2):
        with open(os.path.join(td_results_dir, experiment1, 'experiment_summary.json')) as f:
            td_summary1 = json.load(f)
        with open(os.path.join(td_results_dir, experiment2, 'experiment_summary.json')) as f:
            td_summary2 = json.load(f)
        experiment = ImageAlignmentExperiment(name='image_alignment_%s_to_%s' % (td_summary1['metadata']['image_key'], td_summary2['metadata']['image_key']),
                                              data_sources={'image_path': td_summary1['data_sources'],
                                                            'baseline_image_path': td_summary2['data_sources'],
                                                            'trunks_points': td_summary1['results']['1']['semantic_trunks'].values(),
                                                            'baseline_trunks_points': td_summary2['results']['1']['semantic_trunks'].values(),
                                                            'markers_locations': all_markers_locations[td_summary1['metadata']['image_key']],
                                                            'baseline_markers_locations': all_markers_locations[td_summary2['metadata']['image_key']]},
                                              working_dir=execution_dir)
        experiment.run(repetitions=1)
        results_df.loc['%s__%s' % (experiment1, experiment2), 'by_trunks/mse'] = experiment.results['by_trunks']['mse']
        results_df.loc['%s__%s' % (experiment1, experiment2), 'by_markers/mse'] = experiment.results['by_markers']['mse']
        results_df.loc['%s__%s' % (experiment1, experiment2), 'by_orb/mse'] = experiment.results['by_orb']['mse']
        results_df.loc['%s__%s' % (experiment1, experiment2), 'by_trunks/ssim'] = experiment.results['by_trunks']['ssim']
        results_df.loc['%s__%s' % (experiment1, experiment2), 'by_markers/ssim'] = experiment.results['by_markers']['ssim']
        results_df.loc['%s__%s' % (experiment1, experiment2), 'by_orb/ssim'] = experiment.results['by_orb']['ssim']
    results_df.to_csv(os.path.join(execution_dir, 'results.csv'))