import os
import json

from framework import utils
from content.experiments.template_matching import TemplateMatchingExperiment
from framework import config
from framework import logger

_logger = logger.get_logger()

#################################################################################################
#                                             CONFIG                                            #
#################################################################################################
description = 'template_matching_colored'
methods = ['TM_CCOEFF']
amcl_experiments_paths = [os.path.join(config.base_results_path, 'amcl', 'apr_basic', '15-08-1_15-53-1'),
                          os.path.join(config.base_results_path, 'amcl', 'apr_basic', '15-08-1_16-55-1'),
                          os.path.join(config.base_results_path, 'amcl', 'apr_basic', '15-08-1_19-04-1'),
                          os.path.join(config.base_results_path, 'amcl', 'apr_basic', '15-53-1_16-55-1'),
                          os.path.join(config.base_results_path, 'amcl', 'apr_basic', '15-53-1_19-04-1'),
                          os.path.join(config.base_results_path, 'amcl', 'apr_basic', '16-55-1_19-04-1')]
verbose_mode = False
downsample_rate = 10
use_canopies_masks = False
#################################################################################################


if __name__ == '__main__':
    execution_dir = utils.create_new_execution_folder(description)

    for amcl_experiments_path in amcl_experiments_paths:
        for amcl_experiment in os.listdir(amcl_experiments_path):
            with open(os.path.join(amcl_experiments_path, amcl_experiment, 'experiment_summary.json')) as f:
                amcl_summary = json.load(f)
            map_image_key = amcl_summary['metadata']['map_image_key']
            localization_image_key = amcl_summary['metadata']['localization_image_key']
            map_image_path = os.path.join(amcl_experiments_path, amcl_experiment, 'image_for_map.jpg')
            localization_image_path = os.path.join(amcl_experiments_path, amcl_experiment, 'aligned_image_for_localization.jpg')
            trajectory = amcl_summary['results']['trajectory']
            map_semantic_trunks = amcl_summary['data_sources']['map_semantic_trunks']
            bounding_box_expand_ratio = amcl_summary['params']['bounding_box_expand_ratio']
            max_scan_distance = amcl_summary['params']['max_distance']
            localization_resolution = amcl_summary['params']['localization_resolution']

            _logger.info('Starting %s' % amcl_experiment)
            experiment = TemplateMatchingExperiment(name='template_matching_%s_to_%s' % (map_image_key, localization_image_key),
                                                    data_sources={'map_image_path': map_image_path,
                                                                  'localization_image_path': localization_image_path,
                                                                  'trajectory': trajectory,
                                                                  'map_semantic_trunks': map_semantic_trunks},
                                                    params={'roi_size': max_scan_distance * 2,
                                                            'bounding_box_expand_ratio': bounding_box_expand_ratio,
                                                            'methods': methods,
                                                            'downsample_rate': downsample_rate,
                                                            'localization_resolution': localization_resolution,
                                                            'use_canopies_masks': use_canopies_masks},
                                                    working_dir=execution_dir,
                                                    metadata={'amcl_experiment_path': os.path.join(amcl_experiments_path, amcl_experiment)})
            experiment.run(repetitions=1, verbose_mode=verbose_mode)
