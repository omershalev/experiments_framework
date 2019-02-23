import os
import itertools

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from computer_vision import trunks_detection
from computer_vision import segmentation
from framework.experiment import Experiment
from framework import viz_utils
from framework import cv_utils
from framework.utils import ExperimentFailure

class TrunksDetectionExperiment(Experiment):

    def clean_env(self):
        pass


    def task(self, **kwargs):

        viz_mode = kwargs.get('viz_mode')
        verbose_mode = kwargs.get('verbose')

        # Read image
        image = cv2.imread(self.data_sources)
        cv2.imwrite(os.path.join(self.repetition_dir, 'image.jpg'), image)
        if viz_mode:
            viz_utils.show_image('image', image)

        # Save contours mask
        _, canopies_mask = segmentation.extract_canopy_contours(image)
        cv2.imwrite(os.path.join(self.repetition_dir, 'canopies_mask.jpg'), canopies_mask)

        # Crop central ROI
        cropped_image_size = int(np.min([image.shape[0], image.shape[1]]) * self.params['crop_ratio'])
        cropped_image, crop_origin, _ = cv_utils.crop_region(image, x_center=image.shape[1] / 2, y_center=image.shape[0] / 2,
                                                             x_pixels=cropped_image_size, y_pixels=cropped_image_size)
        _, cropped_canopies_mask = segmentation.extract_canopy_contours(cropped_image)
        crop_square_image = image.copy()
        cv2.rectangle(crop_square_image, crop_origin, (crop_origin[0] + cropped_image_size, crop_origin[1] + cropped_image_size),
                      color=(120, 0, 0), thickness=20)
        cv2.imwrite(os.path.join(self.repetition_dir, 'crop_square_image.jpg'), crop_square_image)
        cv2.imwrite(os.path.join(self.repetition_dir, 'cropped_image.jpg'), cropped_image)
        if viz_mode:
            viz_utils.show_image('cropped image', cropped_image)

        # Estimate orchard orientation
        orientation, angle_to_minima_mean, angle_to_sum_vector = trunks_detection.estimate_rows_orientation(cropped_image)
        rotation_mat = cv2.getRotationMatrix2D((cropped_image.shape[1] / 2, cropped_image.shape[0] / 2), orientation * (-1), scale=1.0)
        vertical_rows_image = cv2.warpAffine(cropped_image, rotation_mat, (cropped_image.shape[1], cropped_image.shape[0]))
        cv2.imwrite(os.path.join(self.repetition_dir, 'vertical_rows.jpg'), vertical_rows_image)
        if verbose_mode:
            angle_to_minima_mean_df = pd.DataFrame(angle_to_minima_mean.values(), index=angle_to_minima_mean.keys(), columns=['minima_mean']).sort_index()
            angle_to_minima_mean_df.to_csv(os.path.join(self.repetition_dir, 'angle_to_minima_mean.csv'))
            self.results[self.repetition_id]['angle_to_minima_mean_path'] = os.path.join(self.repetition_dir, 'angle_to_minima_mean.csv')
            max_sum_value = max(map(lambda vector: vector.max(), angle_to_sum_vector.values()))
            os.mkdir(os.path.join(self.repetition_dir, 'orientation_estimation'))
            for angle in angle_to_sum_vector:
                plt.figure()
                plt.plot(angle_to_sum_vector[angle], color='green')
                plt.xlabel('x')
                plt.ylabel('column sums')
                plt.ylim([(-0.05 * max_sum_value), int(max_sum_value * 1.05)])
                plt.ticklabel_format(axis='y', style='sci', scilimits=(0, 4))
                plt.autoscale(enable=True, axis='x', tight=True)
                plt.tight_layout()
                plt.savefig(os.path.join(self.repetition_dir, 'orientation_estimation', 'sums_vector_%.2f[deg].jpg' % angle))
                rotation_mat = cv2.getRotationMatrix2D((cropped_canopies_mask.shape[1] / 2, cropped_canopies_mask.shape[0] / 2), angle, scale=1.0)
                rotated_canopies_mask = cv2.warpAffine(cropped_canopies_mask, rotation_mat, (cropped_canopies_mask.shape[1], cropped_canopies_mask.shape[0]))
                cv2.imwrite(os.path.join(self.repetition_dir, 'orientation_estimation', 'rotated_canopies_mask_%.2f[deg]_minima_mean=%.2f.jpg'
                                         % (angle, angle_to_minima_mean[angle])), rotated_canopies_mask)
        if viz_mode:
            viz_utils.show_image('vertical rows', vertical_rows_image)

        # Get tree centroids
        centroids, rotated_centroids, aisle_centers, slices_sum_vectors_and_trees, column_sums_vector = trunks_detection.find_tree_centroids(cropped_image, correction_angle=orientation * (-1))
        _, vertical_rows_canopies_mask = segmentation.extract_canopy_contours(vertical_rows_image)
        vertical_rows_aisle_centers_image = cv_utils.draw_lines_on_image(cv2.cvtColor(vertical_rows_canopies_mask, cv2.COLOR_GRAY2BGR),
                                                                         lines_list=[((center, 0), (center, vertical_rows_image.shape[0]))
                                                                         for center in aisle_centers], color=(0, 0, 255))
        slice_image, slice_row_sums_vector, tree_locations_in_row = slices_sum_vectors_and_trees[len(slices_sum_vectors_and_trees) / 2]
        tree_locations = [(slice_image.shape[1] / 2, vertical_location) for vertical_location in tree_locations_in_row]
        slice_image = cv_utils.draw_points_on_image(cv2.cvtColor(slice_image, cv2.COLOR_GRAY2BGR), tree_locations, color=(0, 0, 255))
        cv2.imwrite(os.path.join(self.repetition_dir, 'vertical_rows_aisle_centers.jpg'), vertical_rows_aisle_centers_image)
        plt.figure()
        plt.plot(column_sums_vector, color='green')
        plt.xlabel('x')
        plt.ylabel('column sums')
        plt.ticklabel_format(axis='y', style='sci', scilimits=(0, 4))
        plt.autoscale(enable=True, axis='x', tight=True)
        plt.tight_layout()
        plt.savefig(os.path.join(self.repetition_dir, 'vertical_rows_column_sums.jpg'))
        cv2.imwrite(os.path.join(self.repetition_dir, 'vertical_row_slice.jpg'), slice_image)
        plt.figure(figsize=(4, 5))
        plt.plot(slice_row_sums_vector[::-1], range(len(slice_row_sums_vector)), color='green')
        plt.xlabel('row sums')
        plt.ylabel('y')
        plt.axes().set_aspect(60)
        plt.ticklabel_format(axis='x', style='sci', scilimits=(0, 4))
        plt.autoscale(enable=True, axis='y', tight=True)
        plt.tight_layout()
        plt.savefig(os.path.join(self.repetition_dir, 'slice_row_sums.jpg'))
        vertical_rows_centroids_image = cv_utils.draw_points_on_image(vertical_rows_image, itertools.chain.from_iterable(rotated_centroids), color=(0, 0, 255))
        cv2.imwrite(os.path.join(self.repetition_dir, 'vertical_rows_centroids.jpg'), vertical_rows_centroids_image)
        if viz_mode:
            viz_utils.show_image('vertical rows aisle centers', vertical_rows_aisle_centers_image)
            viz_utils.show_image('vertical rows centroids', vertical_rows_centroids_image)

        # Estimate grid parameters
        grid_dim_x, grid_dim_y = trunks_detection.estimate_grid_dimensions(rotated_centroids)
        shear, drift_vectors, drift_vectors_filtered = trunks_detection.estimate_shear(rotated_centroids)
        drift_vectors_image = cv_utils.draw_lines_on_image(vertical_rows_centroids_image, drift_vectors, color=(255, 255, 0), arrowed=True)
        cv2.imwrite(os.path.join(self.repetition_dir, 'drift_vectors.jpg'), drift_vectors_image)
        drift_vectors_filtered_image = cv_utils.draw_lines_on_image(vertical_rows_centroids_image, drift_vectors_filtered, color=(255, 255, 0), arrowed=True)
        cv2.imwrite(os.path.join(self.repetition_dir, 'drift_vectors_filtered.jpg'), drift_vectors_filtered_image)
        if viz_mode:
            viz_utils.show_image('drift vectors', drift_vectors_filtered_image)

        # Get essential grid
        essential_grid = trunks_detection.get_essential_grid(grid_dim_x, grid_dim_y, shear, orientation, n=self.params['grid_size_for_optimization'])
        essential_grid_shape = np.max(essential_grid, axis=0) - np.min(essential_grid, axis=0)
        margin = essential_grid_shape * 0.2
        essential_grid_shifted = [tuple(elem) for elem in np.array(essential_grid) - np.min(essential_grid, axis=0) + margin / 2]
        estimated_grid_image = np.full((int(essential_grid_shape[1] + margin[1]), int(essential_grid_shape[0] + margin[0]), 3), 255, dtype=np.uint8)
        estimated_grid_image = cv_utils.draw_points_on_image(estimated_grid_image, essential_grid_shifted, color=(255, 0, 0))
        cv2.imwrite(os.path.join(self.repetition_dir, 'estimated_grid.jpg'), estimated_grid_image)
        if viz_mode:
            viz_utils.show_image('estimated grid', estimated_grid_image)

        # Find translation of the grid
        positioned_grid, translation, drift_vectors = trunks_detection.find_min_mse_position(centroids, essential_grid, cropped_image.shape[1], cropped_image.shape[0])
        if positioned_grid is None:
            raise ExperimentFailure
        positioned_grid_image = cv_utils.draw_points_on_image(cropped_image, positioned_grid, color=(255, 0, 0), radius=25)
        positioned_grid_image = cv_utils.draw_points_on_image(positioned_grid_image, centroids, color=(0, 0, 255))
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
        filter_output = np.multiply(gaussians_filter, cropped_canopies_mask)
        cv2.imwrite(os.path.join(self.repetition_dir, 'filter_output.jpg'), filter_output)
        if viz_mode:
            viz_utils.show_image('gaussians filter', gaussians_filter)
            viz_utils.show_image('filter output', filter_output)

        # Optimize the squared grid
        optimized_grid, optimized_grid_args, optimization_steps = trunks_detection.optimize_grid(grid_dim_x, grid_dim_y,
                                                                                                 translation, orientation,
                                                                                                 shear, sigma,
                                                                                                 cropped_image,
                                                                                                 pattern=np.ones([self.params['grid_size_for_optimization'],self.params['grid_size_for_optimization']]))
        optimized_grid_dim_x, optimized_grid_dim_y, optimized_translation_x, optimized_translation_y, optimized_orientation, optimized_shear, optimized_sigma = optimized_grid_args
        self.results[self.repetition_id] = {'optimized_grid_dim_x': optimized_grid_dim_x,
                                            'optimized_grid_dim_y': optimized_grid_dim_y,
                                            'optimized_translation_x': optimized_translation_x,
                                            'optimized_translation_y': optimized_translation_y,
                                            'optimized_orientation': optimized_orientation,
                                            'optimized_shear': optimized_shear,
                                            'optimized_sigma': optimized_sigma}
        optimized_grid_image = cv_utils.draw_points_on_image(cropped_image, optimized_grid, color=(0, 255, 0))
        optimized_grid_image = cv_utils.draw_points_on_image(optimized_grid_image, positioned_grid, color=(255, 0, 0))
        cv2.imwrite(os.path.join(self.repetition_dir, 'optimized_square_grid.jpg'), optimized_grid_image)
        if verbose_mode:
            os.mkdir(os.path.join(self.repetition_dir, 'nelder_mead_steps'))
            self.results[self.repetition_id]['optimization_steps_scores'] = {}
            for step_idx, (step_grid, step_score, step_sigma) in enumerate(optimization_steps):
                self.results[self.repetition_id]['optimization_steps_scores'][step_idx] = step_score
                step_image = cropped_image.copy()
                step_gaussians_filter = trunks_detection.get_gaussians_grid_image(step_grid, step_sigma, cropped_image.shape[1], cropped_image.shape[0])
                step_gaussians_filter = cv2.cvtColor((255.0 * step_gaussians_filter).astype(np.uint8), cv2.COLOR_GRAY2BGR)
                alpha = 0.5
                weighted = cv2.addWeighted(step_image, alpha, step_gaussians_filter, 1 - alpha, gamma=0)
                update_indices = np.where(step_gaussians_filter != 0)
                step_image[update_indices] = weighted[update_indices]
                step_image = cv_utils.draw_points_on_image(step_image, step_grid, color=(0, 255, 0))
                cv2.imwrite(os.path.join(self.repetition_dir, 'nelder_mead_steps', 'optimization_step_%d_[%.2f].jpg' % (step_idx, step_score)), step_image)
        if viz_mode:
            viz_utils.show_image('optimized square grid', optimized_grid_image)

        # Extrapolate full grid on the entire image
        full_grid_np = trunks_detection.extrapolate_full_grid(optimized_grid_dim_x, optimized_grid_dim_y, optimized_orientation, optimized_shear,
                                                              base_grid_origin=np.array(optimized_grid[0]) + np.array(crop_origin),
                                                              image_width=image.shape[1], image_height=image.shape[0])
        full_grid_image = cv_utils.draw_points_on_image(image, [elem for elem in full_grid_np.flatten() if type(elem) is tuple], color=(0, 255, 0))
        cv2.imwrite(os.path.join(self.repetition_dir, 'full_grid.jpg'), full_grid_image)
        if viz_mode:
            viz_utils.show_image('full grid', full_grid_image)

        # Match given orchard pattern to grid
        full_grid_scores_np, full_grid_pose_to_score = trunks_detection.get_grid_scores_array(full_grid_np, image, sigma)
        full_grid_with_scores_image = full_grid_image.copy()
        top_bottom_margin_size = int(0.05 * full_grid_with_scores_image.shape[0])
        left_right_marign_size = int(0.05 * full_grid_with_scores_image.shape[1])
        full_grid_with_scores_image = cv2.copyMakeBorder(full_grid_with_scores_image, top_bottom_margin_size, top_bottom_margin_size,
                                                         left_right_marign_size, left_right_marign_size, cv2.BORDER_CONSTANT,
                                                         dst=None, value=(255, 255, 255))
        for pose, score in full_grid_pose_to_score.items():
            pose = tuple(np.array(pose) + np.array([left_right_marign_size, top_bottom_margin_size]))
            full_grid_with_scores_image = cv_utils.put_shaded_text_on_image(full_grid_with_scores_image, '%.2f' % score,
                                                                            pose, color=(0, 255, 0), offset=(15, 15))
        cv2.imwrite(os.path.join(self.repetition_dir, 'full_grid_with_scores.jpg'), full_grid_with_scores_image)
        orchard_pattern_np = self.params['orchard_pattern']
        pattern_origin, origin_to_sub_scores_array = trunks_detection.fit_pattern_on_grid(full_grid_scores_np, orchard_pattern_np)
        if pattern_origin is None:
            raise ExperimentFailure
        if verbose_mode:
            os.mkdir(os.path.join(self.repetition_dir, 'pattern_matching'))
            for step_origin, step_sub_score_array in origin_to_sub_scores_array.items():
                pattern_matching_image = image.copy()
                step_trunk_coordinates_np = full_grid_np[step_origin[0] : step_origin[0] + orchard_pattern_np.shape[0],
                                                         step_origin[1] : step_origin[1] + orchard_pattern_np.shape[1]]
                step_trunk_points_list = step_trunk_coordinates_np.flatten().tolist()
                pattern_matching_image = cv_utils.draw_points_on_image(pattern_matching_image, step_trunk_points_list, color=(255, 255, 255), radius=25)
                for i in range(step_trunk_coordinates_np.shape[0]):
                    for j in range(step_trunk_coordinates_np.shape[1]):
                        step_trunk_coordinates = (int(step_trunk_coordinates_np[(i, j)][0]), int(step_trunk_coordinates_np[(i, j)][1]))
                        pattern_matching_image = cv_utils.put_shaded_text_on_image(pattern_matching_image, '%.2f' % step_sub_score_array[(i, j)],
                                                                                   step_trunk_coordinates, color=(255, 255, 255), offset=(20, 20))
                pattern_matching_image = cv_utils.draw_points_on_image(pattern_matching_image, [elem for elem in full_grid_np.flatten() if type(elem) is tuple], color=(0, 255, 0))
                mean_score = float(np.mean(step_sub_score_array))
                cv2.imwrite(os.path.join(self.repetition_dir, 'pattern_matching', 'origin=%d_%d_score=%.2f.jpg' %
                                         (step_origin[0], step_origin[1], mean_score)), pattern_matching_image)
        trunk_coordinates_np = full_grid_np[pattern_origin[0] : pattern_origin[0] + orchard_pattern_np.shape[0],
                                            pattern_origin[1] : pattern_origin[1] + orchard_pattern_np.shape[1]]
        trunk_points_list = trunk_coordinates_np[orchard_pattern_np == 1]
        trunk_coordinates_orig_np = trunk_coordinates_np.copy()
        trunk_coordinates_np[orchard_pattern_np != 1] = np.nan
        semantic_trunks_image = cv_utils.draw_points_on_image(image, trunk_points_list, color=(255, 255, 255))
        for i in range(trunk_coordinates_np.shape[0]):
            for j in range(trunk_coordinates_np.shape[1]):
                if np.any(np.isnan(trunk_coordinates_np[(i, j)])):
                    continue
                trunk_coordinates = (int(trunk_coordinates_np[(i, j)][0]), int(trunk_coordinates_np[(i, j)][1]))
                tree_label = '%d/%s' % (j + 1, chr(65 + (trunk_coordinates_np.shape[0] - 1 - i)))
                semantic_trunks_image = cv_utils.put_shaded_text_on_image(semantic_trunks_image, tree_label, trunk_coordinates,
                                                                                  color=(255, 255, 255), offset=(15, 15))
        cv2.imwrite(os.path.join(self.repetition_dir, 'semantic_trunks.jpg'), semantic_trunks_image)
        if viz_mode:
            viz_utils.show_image('semantic trunks', semantic_trunks_image)

        # Refine trunk locations
        refined_trunk_coordinates_np = trunks_detection.refine_trunk_locations(image, trunk_coordinates_np, optimized_sigma,
                                                                               optimized_grid_dim_x, optimized_grid_dim_x)
        confidence = trunks_detection.get_trees_confidence(canopies_mask, refined_trunk_coordinates_np[orchard_pattern_np == 1],
                                                           trunk_coordinates_orig_np[orchard_pattern_np == -1], optimized_sigma)
        refined_trunk_points_list = refined_trunk_coordinates_np[orchard_pattern_np == 1]
        refined_trunk_coordinates_np[orchard_pattern_np != 1] = np.nan
        refined_semantic_trunks_image = cv_utils.draw_points_on_image(image, refined_trunk_points_list, color=(255, 255, 255))
        semantic_trunks = {}
        for i in range(refined_trunk_coordinates_np.shape[0]):
            for j in range(refined_trunk_coordinates_np.shape[1]):
                if np.any(np.isnan(refined_trunk_coordinates_np[(i, j)])):
                    continue
                trunk_coordinates = (int(refined_trunk_coordinates_np[(i, j)][0]), int(refined_trunk_coordinates_np[(i, j)][1]))
                semantic_trunks['%d/%s' % (j + 1, chr(65 + (trunk_coordinates_np.shape[0] - 1 - i)))] = trunk_coordinates
                tree_label = '%d/%s' % (j + 1, chr(65 + (refined_trunk_coordinates_np.shape[0] - 1 - i)))
                refined_semantic_trunks_image = cv_utils.put_shaded_text_on_image(refined_semantic_trunks_image, tree_label, trunk_coordinates,
                                                                                  color=(255, 255, 255), offset=(15, 15))
        tree_scores_stats = trunks_detection.get_tree_scores_stats(canopies_mask, trunk_points_list, optimized_sigma)
        self.results[self.repetition_id]['semantic_trunks'] = semantic_trunks
        self.results[self.repetition_id]['tree_scores_stats'] = tree_scores_stats
        self.results[self.repetition_id]['confidence'] = confidence
        cv2.imwrite(os.path.join(self.repetition_dir, 'refined_semantic_trunks[%.2f].jpg' % confidence), refined_semantic_trunks_image)
        if viz_mode:
            viz_utils.show_image('refined semantic trunks', refined_semantic_trunks_image)

