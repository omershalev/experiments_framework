import json
import cv2
import numpy as np
import os

from content.data_pointers.lavi_april_18.dji import snapshots_80_meters, snapshots_80_meters_markers_locations_json_path
from framework import cv_utils
from framework import viz_utils
from framework.experiment import Experiment


class TemplateMatchingExperiment(Experiment):

    def clean_env(self):
        pass

    def task(self, **kwargs):

        viz_mode = kwargs.get('viz_mode')

        map_image_path = self.data_sources['map_image_path']
        localization_image_path = self.data_sources['localization_image_path']
        roi_center = self.data_sources['roi_center']
        map_alignment_points = self.data_sources['map_alignment_points']
        localization_alignment_points = self.data_sources['localization_alignment_points']
        roi_size = self.params['roi_size']

        map_image = cv2.imread(map_image_path)
        localization_image = cv2.imread(localization_image_path)
        localization_image, _ = cv_utils.warp_image(localization_image, localization_alignment_points, map_alignment_points)
        roi_image, _, _ = cv_utils.crop_region(localization_image, roi_center[0], roi_center[1], roi_size, roi_size)
        matches_image = map_image.copy()
        cv2.circle(matches_image, roi_center, radius=15, color=(0, 0, 255), thickness=-1)
        cv2.rectangle(matches_image, (roi_center[0] - roi_size / 2, roi_center[1] - roi_size / 2),
                                     (roi_center[0] + roi_size / 2, roi_center[1] + roi_size / 2), (0, 0, 255), thickness=2)

        for method in ['TM_CCOEFF', 'TM_CCOEFF_NORMED', 'TM_CCORR', 'TM_CCORR_NORMED', 'TM_SQDIFF', 'TM_SQDIFF_NORMED']:
            matching_result = cv2.matchTemplate(map_image, roi_image, method=eval('cv2.%s' % method))
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(matching_result)
            if method in ['TM_SQDIFF', 'TM_SQDIFF_NORMED']:
                match_top_left = min_loc
            else:
                match_top_left = max_loc
            match_bottom_right = (match_top_left[0] + roi_image.shape[1], match_top_left[1] + roi_image.shape[0])
            match_center = (match_top_left[0] + roi_image.shape[1] / 2, match_top_left[1] + roi_image.shape[0] / 2)
            cv2.rectangle(matches_image, match_top_left, match_bottom_right, (255, 0, 0), thickness=2)
            cv2.circle(matches_image, match_center, radius=15, color=(255, 0, 0), thickness=-1)
            cv2.imwrite(os.path.join(self.repetition_dir, 'matches.jpg'), matches_image)
        if viz_mode:
            viz_utils.show_image('matches image', matches_image)

        self.results[self.repetition_id]['error'] = np.sqrt((roi_center[0] - match_center[0]) ** 2 + (roi_center[1] - match_center[1]) ** 2)


if __name__ == '__main__':
    image_key1 = snapshots_80_meters.keys()[0]
    # image_key2 = snapshots_80_meters.keys()[2]
    image_key2 = image_key1
    with open(snapshots_80_meters_markers_locations_json_path) as f:
        all_markers_locations = json.load(f)
    points1 = all_markers_locations[image_key1]
    points2 = all_markers_locations[image_key2]


    experiment = TemplateMatchingExperiment(name='template_matching_%s_to_%s' % (image_key1, image_key2),
                                            data_sources={'map_image_path': snapshots_80_meters[image_key1].path,
                                                          'localization_image_path': snapshots_80_meters[image_key2].path,
                                                          'roi_center': (1900, 2000), # TODO: change!!!
                                                          'map_alignment_points': points1,
                                                          'localization_alignment_points': points2},
                                            params={'roi_size': 350}, # TODO: change!!! (you can play with this parameter and show the different results but everything still doesn't help...)
                                            working_dir=r'/home/omer/temp', # TODO: change!!!
                                            metadata={}# TODO: change!!!
                                            )
    experiment.run(repetitions=1, viz_mode=True)

    # TODO: align images
    # TODO: show also how it works on grayscale and on contours mask (i.e. not working) ==> something temporal is needed (and in lower dimension, i.e. laser)
    # TODO: The poent here is that template matching is difficult with orchard image (everything's green and brown) ==> even temporal thing (LSTM) will be difficult because you need some basic recognition to work!!