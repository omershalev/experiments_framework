import os
import itertools

import cv2
import numpy as np
import matplotlib.pyplot as plt

from computer_vision import trunks_detection
from computer_vision import segmentation
from framework.experiment import Experiment
from framework import viz_utils
from framework import cv_utils


class TrunksDetectionExperiment(Experiment):

    def clean_env(self):
        pass


    def task(self, **kwargs):

        viz_mode = kwargs.get('viz_mode')

        # Read image
        image = cv2.imread(self.data_sources)
        cv2.imwrite(os.path.join(self.repetition_dir, 'image.jpg'), image)
        if viz_mode:
            viz_utils.show_image('image', image)

        # Save contours mask
        _, contours_mask = segmentation.extract_canopy_contours(image)
        cv2.imwrite(os.path.join(self.repetition_dir, 'contours_mask.jpg'), contours_mask)

        # Crop central ROI
        cropped_image_size = np.min([image.shape[0], image.shape[1]]) * self.params['crop_ratio']
        cropped_image, crop_origin, _ = cv_utils.crop_region(image, x_center=image.shape[1] / 2, y_center=image.shape[0] / 2,
                                                             x_pixels=cropped_image_size, y_pixels=cropped_image_size)
        cv2.imwrite(os.path.join(self.repetition_dir, 'cropped_image.jpg'), cropped_image)
        if viz_mode:
            viz_utils.show_image('cropped image', cropped_image)

        # Estimate orchard orientation
        orientation = trunks_detection.estimate_rows_orientation(cropped_image)
        rotation_mat = cv2.getRotationMatrix2D((cropped_image.shape[1] / 2, cropped_image.shape[0] / 2), orientation * (-1), scale=1.0)
        vertical_rows_image = cv2.warpAffine(cropped_image, rotation_mat, (cropped_image.shape[1], cropped_image.shape[0]))
        cv2.imwrite(os.path.join(self.repetition_dir, 'vertical_rows.jpg'), vertical_rows_image)
        if viz_mode:
            viz_utils.show_image('vertical rows', vertical_rows_image)

        # Get tree centroids
        centroids, rotated_centroids, aisle_centers, slices_and_cumsums = trunks_detection.find_tree_centroids(cropped_image, correction_angle=orientation * (-1))
        vertical_rows_aisle_centers_image = cv_utils.draw_lines_on_image(vertical_rows_image, lines_list=[((center, 0), (center, vertical_rows_image.shape[0]))
                                                                         for center in aisle_centers], color=(0, 0, 255))
        cv2.imwrite(os.path.join(self.repetition_dir, 'vertical_rows_aisle_centers.jpg'), vertical_rows_aisle_centers_image)
        slice_image, cumsum_vector = slices_and_cumsums[len(slices_and_cumsums) / 2]
        cv2.imwrite(os.path.join(self.repetition_dir, 'vertical_row_slice.jpg'), slice_image)
        fig = plt.figure()
        plt.plot(cumsum_vector)
        plt.savefig(os.path.join(self.repetition_dir, 'cumsum_vector.jpg'))
        vertical_rows_centroids_image = cv_utils.draw_points_on_image(vertical_rows_image, itertools.chain.from_iterable(rotated_centroids), color=(0, 0, 255))
        cv2.imwrite(os.path.join(self.repetition_dir, 'vertical_rows_centroids.jpg'), vertical_rows_centroids_image)
        if viz_mode:
            viz_utils.show_image('vertical rows aisle centers', vertical_rows_aisle_centers_image)
            viz_utils.show_image('vertical rows centroids', vertical_rows_centroids_image)

        # Estimate grid parameters
        grid_dim_x, grid_dim_y = trunks_detection.estimate_grid_dimensions(rotated_centroids)
        shear, drift_vectors = trunks_detection.estimate_shear(rotated_centroids)
        drift_vectors_image = cv_utils.draw_lines_on_image(vertical_rows_centroids_image, drift_vectors, color=(255, 255, 0))
        cv2.imwrite(os.path.join(self.repetition_dir, 'drift_vectors.jpg'), drift_vectors_image)
        if viz_mode:
            viz_utils.show_image('drift vectors', drift_vectors_image)

        # Get essential grid
        essential_grid = trunks_detection.get_essential_grid(grid_dim_x, grid_dim_y, shear, orientation, n=self.params['grid_size_for_optimization'])
        essential_grid_shape = np.max(essential_grid, axis=0) - np.min(essential_grid, axis=0)
        margin = essential_grid_shape * 0.2
        essential_grid_shifted = [tuple(elem) for elem in np.array(essential_grid) - np.min(essential_grid, axis=0) + margin / 2]
        estimated_grid_image = np.full((int(essential_grid_shape[1] + margin[1]), int(essential_grid_shape[0] + margin[0]), 3), 0, dtype=np.uint8)
        estimated_grid_image = cv_utils.draw_points_on_image(estimated_grid_image, essential_grid_shifted, color=(255, 0, 0))
        cv2.imwrite(os.path.join(self.repetition_dir, 'estimated_grid.jpg'), estimated_grid_image)
        if viz_mode:
            viz_utils.show_image('estimated grid', estimated_grid_image)

        # Find translation of the grid
        positioned_grid, translation, drift_vectors = trunks_detection.find_min_mse_position(centroids, essential_grid, cropped_image.shape[1], cropped_image.shape[0])
        positioned_grid_image = cv_utils.draw_points_on_image(cropped_image, positioned_grid, color=(255, 0, 0), radius=20)
        positioned_grid_image = cv_utils.draw_points_on_image(positioned_grid_image, centroids, color=(0, 0, 255), radius=10)
        positioned_grid_image = cv_utils.draw_lines_on_image(positioned_grid_image, drift_vectors, color=(255, 255, 0), thickness=3)
        cv2.imwrite(os.path.join(self.repetition_dir, 'positioned_grid.jpg'), positioned_grid_image)
        if viz_mode:
            viz_utils.show_image('positioned grid', positioned_grid_image)

        # Estimate sigma as a portion of intra-row distance
        sigma = grid_dim_y * self.params['initial_sigma_to_dim_y_ratio']

        # Get a grid of gaussians
        grid = trunks_detection.get_grid(grid_dim_x, grid_dim_y, translation, orientation, shear, n=self.params['grid_size_for_optimization'])
        gaussians_filter = trunks_detection.get_gaussians_grid_image(grid, sigma, cropped_image.shape[1], cropped_image.shape[0])
        cv2.imwrite(os.path.join(self.repetition_dir, 'gaussians_filter.jpg'), 255.0 * gaussians_filter)
        _, contours_mask = segmentation.extract_canopy_contours(cropped_image)
        filter_output = np.multiply(gaussians_filter, contours_mask)
        cv2.imwrite(os.path.join(self.repetition_dir, 'filter_output.jpg'), filter_output)
        if viz_mode:
            viz_utils.show_image('gaussians filter', gaussians_filter)
            viz_utils.show_image('filter output', filter_output)
        # TODO: if 7x7 is not feasible, consider taking a 7x6 or 8x6 grid - this might help...

        # Optimize the grid
        optimized_grid, optimized_grid_args = trunks_detection.optimize_grid(grid_dim_x, grid_dim_y, translation, orientation, shear, sigma, cropped_image, n=self.params['grid_size_for_optimization'])
        optimized_grid_dim_x, optimized_grid_dim_y, optimized_translation_x, optimized_translation_y, optimized_orientation, optimized_shear, optimized_sigma = optimized_grid_args
        optimized_grid_image = cv_utils.draw_points_on_image(cropped_image, optimized_grid, color=(0, 255, 0))
        optimized_grid_image = cv_utils.draw_points_on_image(optimized_grid_image, positioned_grid, color=(255, 0, 0))
        cv2.imwrite(os.path.join(self.repetition_dir, 'optimized_grid.jpg'), optimized_grid_image)
        if viz_mode:
            viz_utils.show_image('optimized grid', optimized_grid_image)

        # Extrapolate full grid on the entire image
        full_grid_np = trunks_detection.extrapolate_full_grid(optimized_grid_dim_x, optimized_grid_dim_y, optimized_orientation, optimized_shear,
                                                           base_grid_origin=np.array(optimized_grid[0]) + np.array(crop_origin),
                                                           image_width=image.shape[1], image_height=image.shape[0])
        full_grid_image = cv_utils.draw_points_on_image(image, [elem for elem in full_grid_np.flatten() if type(elem) is tuple], color=(255, 0, 0))
        cv2.imwrite(os.path.join(self.repetition_dir, 'full_grid.jpg'), full_grid_image)
        if viz_mode:
            viz_utils.show_image('full grid', full_grid_image)


        # Match given pattern to grid
        scores_array_np = trunks_detection.get_grid_scores_array(full_grid_np, image, optimized_sigma)
        pattern_np = self.params['pattern']
        pattern_origin = trunks_detection.fit_pattern_on_grid(scores_array_np, pattern_np)
        pattern_coordinates_np = full_grid_np[pattern_origin[0] : pattern_origin[0] + pattern_np.shape[0],
                                           pattern_origin[1] : pattern_origin[1] + pattern_np.shape[1]]
        pattern_points = pattern_coordinates_np[pattern_np != -1]
        semantic_image = cv_utils.draw_points_on_image(image, pattern_points, color=(255, 255, 255))
        for i in range(pattern_coordinates_np.shape[0]):
            for j in range(pattern_coordinates_np.shape[1]):
                if pattern_np[(i, j)] == -1:
                    continue
                trunk_coordinates = (int(pattern_coordinates_np[(i, j)][0]) + 15, int(pattern_coordinates_np[(i, j)][1]) + 15)
                tree_label = '%d/%s' % (j + 1, chr(65 + (pattern_coordinates_np.shape[0] - 1 - i)))
                cv2.putText(semantic_image, tree_label, trunk_coordinates, fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                            fontScale=2, color=(255, 255, 255), thickness=8, lineType=cv2.LINE_AA)
        cv2.imwrite(os.path.join(self.repetition_dir, 'semantic_image.jpg'), semantic_image)
        if viz_mode:
            viz_utils.show_image('semantic', semantic_image)


if __name__ == '__main__':

    from content.data_pointers.lavi_april_18 import dji

    image_key = dji.snapshots_60_meters.keys()[0]
    data_descriptor = dji.snapshots_60_meters[image_key]
    pattern_np = np.ones((9, 10), dtype=np.int8)
    pattern_np[0:5, 0] = -1

    experiment = TrunksDetectionExperiment(name='trunks detection on %s' % image_key, data_sources=data_descriptor.path, working_dir=r'/home/omer/temp',
                                           params={'crop_ratio': 1.0, 'initial_sigma_to_dim_y_ratio': 0.33, 'grid_size_for_optimization': 6,
                                                   'pattern': pattern_np})
    experiment.run(repetitions=5, viz_mode=False)