import os
import json

from framework import utils
from content.experiments.template_matching import TemplateMatchingExperiment


#################################################################################################
#                                             CONFIG                                            #
#################################################################################################
methods = ['TM_CCOEFF', 'TM_CCOEFF_NORMED', 'TM_CCORR', 'TM_CCORR_NORMED', 'TM_SQDIFF', 'TM_SQDIFF_NORMED']
amcl_experiment_path = r'/home/omer/orchards_ws/results/amcl/20190210-000411_amcl_baseline_different_source_and_target/20190210-000411_amcl_snapshots_for_s_patrol_trajectory_on_15-08-1_and_15-53-1'
verbose_mode = False
downsample_rate = 10
#################################################################################################


if __name__ == '__main__':
    execution_dir = utils.create_new_execution_folder('template_matching')

    with open(os.path.join(amcl_experiment_path, 'experiment_summary.json')) as f:
        amcl_summary = json.load(f)
    map_image_key = amcl_summary['metadata']['map_image_key']
    localization_image_key = amcl_summary['metadata']['localization_image_key']
    map_image_path = amcl_summary['data_sources']['map_image_path']
    localization_image_path = os.path.join(amcl_experiment_path, 'aligned_image_for_localization.jpg')
    trajectory = amcl_summary['results']['trajectory']
    map_semantic_trunks = amcl_summary['data_sources']['map_semantic_trunks']
    bounding_box_expand_ratio = amcl_summary['params']['bounding_box_expand_ratio']
    max_scan_distance = amcl_summary['params']['max_distance']
    localization_resolution = amcl_summary['params']['localization_resolution']

    experiment = TemplateMatchingExperiment(name='template_matching_%s_to_%s' % (map_image_key, localization_image_key),
                                            data_sources={'map_image_path': map_image_path,
                                                          'localization_image_path': localization_image_path,
                                                          'trajectory': trajectory,
                                                          'map_semantic_trunks': map_semantic_trunks},
                                            params={'roi_size': max_scan_distance * 2,
                                                    'bounding_box_expand_ratio': bounding_box_expand_ratio,
                                                    'methods': methods,
                                                    'downsample_rate': downsample_rate,
                                                    'localization_resolution': localization_resolution},
                                            working_dir=execution_dir,
                                            metadata={'amcl_experiment_path': amcl_experiment_path})
    experiment.run(repetitions=1, verbose_mode=verbose_mode)
