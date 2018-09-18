import numpy as np
import cv2
import itertools

from nelder_mead import NelderMead

from computer_vision import segmentation
from computer_vision import trunks_detection
from framework import cv_utils
from framework import viz_utils
import content.data_pointers.lavi_april_18.dji as dji_data


image_paths_list = [descriptor.path for descriptor in dji_data.snapshots_60_meters.values() + dji_data.snapshots_80_meters.values()]
viz_mode = True
N = 6

if __name__ == '__main__':
    idx = 0
    for image_path in image_paths_list:

        idx += 1
        if idx != 4:
            continue
        # Read image
        image = cv2.imread(image_path)
        if viz_mode:
            viz_utils.show_image('image', image)

        # Crop central ROI
        cropped_image_size = np.min([image.shape[0], image.shape[1]]) * 0.9
        cropped_image, crop_origin, _ = cv_utils.crop_region(image, x_center=image.shape[1] / 2, y_center=image.shape[0] / 2,
                                                            x_pixels=cropped_image_size, y_pixels=cropped_image_size)
        if viz_mode:
            viz_utils.show_image('cropped image', cropped_image)

        # Estimate orchard orientation
        orientation = trunks_detection.estimate_rows_orientation(cropped_image)
        if viz_mode:
            rotation_mat = cv2.getRotationMatrix2D((cropped_image.shape[1] / 2, cropped_image.shape[0] / 2), orientation * (-1), scale=1.0) # TODO: center point ? (+ coordinates order)
            vertical_rows_image = cv2.warpAffine(cropped_image, rotation_mat, (cropped_image.shape[1], cropped_image.shape[0]))
            viz_utils.show_image('vertical rows', vertical_rows_image)

        # Get tree centroids
        centroids, rotated_centroids = trunks_detection.find_tree_centroids(cropped_image, correction_angle=orientation * (-1))
        if viz_mode:
            # TODO: visualize the cum sum graphs
            vertical_rows_centroids_image = cv_utils.draw_points_on_image(vertical_rows_image, itertools.chain.from_iterable(rotated_centroids), color=(0, 0, 255))
            viz_utils.show_image('vertical rows centroids', vertical_rows_centroids_image)
            # centroids_image = cv_utils.draw_points_on_image(cropped_image, centroids, color=(0, 0, 255))
            # viz_utils.show_image('centroids', centroids_image)

        # Estimate grid parameters
        grid_dim_x, grid_dim_y = trunks_detection.estimate_grid_dimensions(rotated_centroids)
        shear, drift_vectors = trunks_detection.estimate_shear(rotated_centroids)
        if viz_mode:
            drift_vectors_image = cv_utils.draw_lines_on_image(vertical_rows_centroids_image, drift_vectors, color=(255, 255, 0))
            viz_utils.show_image('drift vectors', drift_vectors_image)

        # Get essential grid
        essential_grid = trunks_detection.get_essential_grid(grid_dim_x, grid_dim_y, shear, orientation, n=N)
        if viz_mode:
            essential_grid_shape = np.max(essential_grid, axis=0) - np.min(essential_grid, axis=0)
            margin = essential_grid_shape * 0.2
            essential_grid_shifted = [tuple(elem) for elem in np.array(essential_grid) - np.min(essential_grid, axis=0) + margin / 2]
            estimated_grid_image = np.full((int(essential_grid_shape[1] + margin[1]), int(essential_grid_shape[0] + margin[0]), 3), 0, dtype=np.uint8)
            estimated_grid_image = cv_utils.draw_points_on_image(estimated_grid_image, essential_grid_shifted, color=(255, 0, 0))
            viz_utils.show_image('estimated grid', estimated_grid_image)

        # Find translation of the grid
        positioned_grid, translation, drift_vectors = trunks_detection.find_min_mse_position(centroids, essential_grid, cropped_image.shape[1], cropped_image.shape[0])
        if viz_mode:
            positioned_grid_image = cv_utils.draw_points_on_image(cropped_image, positioned_grid, color=(255, 0, 0), radius=20)
            positioned_grid_image = cv_utils.draw_points_on_image(positioned_grid_image, centroids, color=(0, 0, 255), radius=10)
            positioned_grid_image = cv_utils.draw_lines_on_image(positioned_grid_image, drift_vectors, color=(255, 255, 0), thickness=3)
            viz_utils.show_image('positioned grid', positioned_grid_image)

        # Estimate sigma to one third of intra-row distance
        sigma = grid_dim_y / 3

        # Get a grid of gaussians
        grid = trunks_detection.get_grid(grid_dim_x, grid_dim_y, translation, orientation, shear, n=N)
        gaussians_filter = trunks_detection.get_gaussians_grid_image(grid, sigma, cropped_image.shape[1], cropped_image.shape[0])
        if viz_mode:
            viz_utils.show_image('gaussians', gaussians_filter)
            _, contours_mask = segmentation.extract_canopy_contours(cropped_image)
            filter_result = np.multiply(gaussians_filter, contours_mask)
            viz_utils.show_image('filter result', filter_result)

        # OPTIMIZATION TODO: improve and arrange
        opt = trunks_detection.TrunksGridOptimization(grid_dim_x, grid_dim_y, translation, orientation, shear, sigma, cropped_image, n=N)
        nm = NelderMead(opt.target, opt.get_params())
        optimized_grid_args, _ = nm.maximize(n_iter=30)
        optimized_grid_dim_x, optimized_grid_dim_y, optimized_translation_x, optimized_translation_y, optimized_orientation, optimized_shear, optimized_sigma = optimized_grid_args
        optimized_grid = trunks_detection.get_grid(optimized_grid_dim_x, optimized_grid_dim_y,
                                                   (optimized_translation_x, optimized_translation_y), optimized_orientation, optimized_shear, n=N)
        if viz_mode:
            optimized_grid_image = cv_utils.draw_points_on_image(cropped_image, optimized_grid, color=(0, 255, 0))
            optimized_grid_image = cv_utils.draw_points_on_image(optimized_grid_image, positioned_grid, color=(255, 0, 0))
            viz_utils.show_image('optimized grid', optimized_grid_image)



        # TODO: title


        full_grid_np = trunks_detection.extrapolate_full_grid(optimized_grid_dim_x, optimized_grid_dim_y, optimized_orientation, optimized_shear,
                                                           base_grid_origin=np.array(optimized_grid[0]) + np.array(crop_origin),
                                                           image_width=image.shape[1], image_height=image.shape[0])

        if viz_mode:
            full_grid_image = cv_utils.draw_points_on_image(image, [elem for elem in full_grid_np.flatten() if type(elem) is tuple], color=(255, 0, 0))
            viz_utils.show_image('full grid', full_grid_image)


        ### WIP area

        scores_array_np = trunks_detection.get_grid_scores_array(full_grid_np, image, optimized_sigma)

        pattern_np = np.ones((9, 10), dtype=np.int8)
        pattern_np[0:5, 0] = -1


        pattern_origin = trunks_detection.fit_pattern_on_grid(scores_array_np, pattern_np)
        pattern_coordinates_np = full_grid_np[pattern_origin[0] : pattern_origin[0] + pattern_np.shape[0],
                                           pattern_origin[1] : pattern_origin[1] + pattern_np.shape[1]]

        if viz_mode:
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

            viz_utils.show_image('semantic', semantic_image)

        # TODO: consider adding one additional line per the bottom isle (the one for cars) of the empty ilse (-1's)

        print ('end of iteration')
        break