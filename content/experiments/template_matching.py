import cv2
import os
import numpy as np
import pandas as pd

from framework import cv_utils
from framework import logger
from framework.experiment import Experiment
from computer_vision import segmentation

MESSAGING_FREQUENCY = 100

_logger = logger.get_logger()


class TemplateMatchingExperiment(Experiment):

    def clean_env(self):
        pass

    def task(self, **kwargs):

        verbose_mode = kwargs.get('verbose_mode')

        # Read params and data sources
        map_image_path = self.data_sources['map_image_path']
        localization_image_path = self.data_sources['localization_image_path']
        trajectory = self.data_sources['trajectory']
        map_semantic_trunks = self.data_sources['map_semantic_trunks']
        bounding_box_expand_ratio = self.params['bounding_box_expand_ratio']
        roi_size = self.params['roi_size']
        methods = self.params['methods']
        downsample_rate = self.params['downsample_rate']
        localization_resolution = self.params['localization_resolution']
        use_canopies_masks = self.params['use_canopies_masks']

        # Read images
        map_image = cv2.imread(map_image_path)
        localization_image = cv2.imread(localization_image_path)
        upper_left, lower_right = cv_utils.get_bounding_box(map_image, map_semantic_trunks.values(), expand_ratio=bounding_box_expand_ratio)
        map_image = map_image[upper_left[1]:lower_right[1], upper_left[0]:lower_right[0]]
        localization_image = localization_image[upper_left[1]:lower_right[1], upper_left[0]:lower_right[0]]
        if use_canopies_masks:
            _, map_image = segmentation.extract_canopy_contours(map_image)
            _, localization_image = segmentation.extract_canopy_contours(localization_image)
        cv2.imwrite(os.path.join(self.experiment_dir, 'map_image.jpg'), map_image)
        cv2.imwrite(os.path.join(self.experiment_dir, 'localization_image.jpg'), localization_image)

        # Initialize errors dataframe
        errors = pd.DataFrame(index=map(lambda point: '%s_%s' % (point[0], point[1]), trajectory), columns=methods)

        # Loop over points in trajectory
        for ugv_pose_idx, ugv_pose in enumerate(trajectory):
            if ugv_pose_idx % downsample_rate != 0:
                continue
            if ugv_pose_idx % MESSAGING_FREQUENCY == 0:
                _logger.info('At point #%d' % ugv_pose_idx)
            roi_image, _, _ = cv_utils.crop_region(localization_image, ugv_pose[0], ugv_pose[1], roi_size, roi_size)
            if verbose_mode:
                matches_image = map_image.copy()
                cv2.circle(matches_image, tuple(ugv_pose), radius=15, color=(0, 0, 255), thickness=-1)
                cv2.rectangle(matches_image, (ugv_pose[0] - roi_size / 2, ugv_pose[1] - roi_size / 2),
                              (ugv_pose[0] + roi_size / 2, ugv_pose[1] + roi_size / 2), (0, 0, 255), thickness=2)
            for method in methods:
                matching_result = cv2.matchTemplate(map_image, roi_image, method=eval('cv2.%s' % method))
                min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(matching_result)
                if method in ['TM_SQDIFF', 'TM_SQDIFF_NORMED']:
                    match_top_left = min_loc
                else:
                    match_top_left = max_loc
                match_bottom_right = (match_top_left[0] + roi_image.shape[1], match_top_left[1] + roi_image.shape[0])
                match_center = (match_top_left[0] + roi_image.shape[1] / 2, match_top_left[1] + roi_image.shape[0] / 2)
                error = np.sqrt((ugv_pose[0] - match_center[0]) ** 2 + (ugv_pose[1] - match_center[1]) ** 2) * localization_resolution
                errors.loc['%s_%s' % (ugv_pose[0], ugv_pose[1]), method] = error
                if verbose_mode:
                    cv2.rectangle(matches_image, match_top_left, match_bottom_right, (255, 0, 0), thickness=2)
                    cv2.circle(matches_image, match_center, radius=15, color=(255, 0, 0), thickness=-1)
                    cv2.imwrite(os.path.join(self.repetition_dir, 'matches_%s_%s.jpg' % (ugv_pose[0], ugv_pose[1])), matches_image)

        # Save results
        errors.to_csv(os.path.join(self.experiment_dir, 'errors.csv'))
