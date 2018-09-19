import os
import json
import cv2

from computer_vision import typical_image_alignment
from framework.experiment import Experiment
from framework import cv_utils


class MapAlignmentExperiment(Experiment):

    def clean_env(self):
        pass


    def task(self, **kwargs):

        # Read image
        image = cv2.imread(self.data_sources['image_path'])
        baseline_image = cv2.imread(self.data_sources['baseline_image_path'])
        cv2.imwrite(os.path.join(self.repetition_dir, 'image.jpg'), image)
        cv2.imwrite(os.path.join(self.repetition_dir, 'baseline_image.jpg'), baseline_image)

        # Align images by markers
        marker_locations = self.data_sources['markers_locations']
        baseline_marker_locations = self.data_sources['baseline_markers_locations']
        warped_image_by_markers = cv_utils.warp_image(image=image, points_in_image=marker_locations, points_in_baseline=baseline_marker_locations)
        cv2.imwrite(os.path.join(self.repetition_dir, 'warped by markers.jpg'), warped_image_by_markers)
        mse = cv_utils.calculate_image_diff(baseline_image, warped_image_by_markers, method='mse')
        ssim = cv_utils.calculate_image_diff(baseline_image, warped_image_by_markers, method='ssim')
        self.results['by_markers'] = {'mse': mse, 'ssim': ssim}

        # Align images by typical flow
        warped_image_by_orb, orb_matches_image = typical_image_alignment.orb_based_registration(image, baseline_image)
        cv2.imwrite(os.path.join(self.repetition_dir, 'warped by orb.jpg'), warped_image_by_orb)
        cv2.imwrite(os.path.join(self.repetition_dir, 'orb matches.jpg'), orb_matches_image)
        mse = cv_utils.calculate_image_diff(baseline_image, warped_image_by_orb, method='mse')
        ssim = cv_utils.calculate_image_diff(baseline_image, warped_image_by_orb, method='ssim')
        self.results['by_orb'] = {'mse': mse, 'ssim': ssim}

        # Align images by trunks points
        trunks = self.data_sources['trunks_points']
        baseline_trunks = self.data_sources['baseline_trunks_points']
        warped_image_by_trunks = cv_utils.warp_image(image=image, points_in_image=trunks, points_in_baseline=baseline_trunks)
        cv2.imwrite(os.path.join(self.repetition_dir, 'warped by trunks.jpg'), warped_image_by_trunks)
        mse = cv_utils.calculate_image_diff(baseline_image, warped_image_by_trunks, method='mse')
        ssim = cv_utils.calculate_image_diff(baseline_image, warped_image_by_trunks, method='ssim')
        self.results['by_trunks'] = {'mse': mse, 'ssim': ssim}


if __name__ == '__main__':

    from content.data_pointers.lavi_april_18 import dji

    with open(r'/home/omer/Downloads/experiment_metadata1.json') as f:
        summary1 = json.load(f)
    with open(r'/home/omer/Downloads/experiment_metadata2.json') as f:
        summary2 = json.load(f)
    with open(dji.snapshots_60_meters_markers_locations_json_path) as f:
        all_markers_locations = json.load(f)

    image_key = '19-03-3'
    baseline_image_key = '16-54-1'

    image_descriptor = dji.snapshots_60_meters[image_key]
    baseline_image_descriptor = dji.snapshots_60_meters[baseline_image_key]
    trunks_points = summary1['results']['1']['pattern_points']
    baseline_trunks_points = summary2['results']['1']['pattern_points']
    marker_locations = all_markers_locations[image_key]
    baseline_marker_locations = all_markers_locations[baseline_image_key]


    experiment = MapAlignmentExperiment(name='map alignment of %s to %s' % (image_key, baseline_image_key),
                                        data_sources={'baseline_image_path': baseline_image_descriptor.path, 'image_path': image_descriptor.path,
                                                      'trunks_points': trunks_points, 'baseline_trunks_points': baseline_trunks_points,
                                                      'markers_locations': marker_locations, 'baseline_markers_locations': baseline_marker_locations},
                                        working_dir=r'/home/omer/temp')
    experiment.run(repetitions=1)