import os
import shutil

from framework import utils
from framework.config import base_results_path

original_execution_dir = 'archive - to be removed/trunks_detection_nov2'

if __name__ == '__main__':
    execution_dir = utils.create_new_execution_folder('trunk_detection_post_processing')
    for experiment_name in os.listdir(os.path.join(base_results_path, 'trunks_detection', original_execution_dir)):
        repetition_dir = os.path.join(base_results_path, 'trunks_detection', original_execution_dir, experiment_name, '1')
        optimized_square_grid_path = os.path.join(repetition_dir, 'optimized_square_grid.jpg')
        semantic_trunks_path = os.path.join(repetition_dir, 'semantic_trunks.jpg')
        contours_mask_path = os.path.join(repetition_dir, 'contours_mask.jpg')
        refined_semantic_trunks_path = os.path.join(repetition_dir, filter(lambda filename: filename.startswith('refined_semantic_trunks'),
                                                                           os.listdir(repetition_dir))[0])
        new_optimized_filename = '%s_%s' % (experiment_name.split('_')[-1], optimized_square_grid_path.split('/')[-1])
        shutil.copyfile(optimized_square_grid_path, os.path.join(execution_dir, new_optimized_filename))
        new_semantic_filename = '%s_%s' % (experiment_name.split('_')[-1], semantic_trunks_path.split('/')[-1])
        shutil.copyfile(semantic_trunks_path, os.path.join(execution_dir, new_semantic_filename))
        new_refinement_filename = '%s_%s' % (experiment_name.split('_')[-1], refined_semantic_trunks_path.split('/')[-1])
        shutil.copyfile(refined_semantic_trunks_path, os.path.join(execution_dir, new_refinement_filename))
        new_contours_mask_filename = '%s_%s' % (experiment_name.split('_')[-1], contours_mask_path.split('/')[-1])
        shutil.copyfile(contours_mask_path, os.path.join(execution_dir, new_contours_mask_filename))
