import numpy as np
import cv2
import itertools

from computer_vision import segmentation
from computer_vision import trunks_detection
from framework import cv_utils
from framework import viz_utils
import content.data_pointers.lavi_april_18.dji as dji_data

image_paths_list = [descriptor.path for descriptor in dji_data.snapshots_60_meters.values() + dji_data.snapshots_80_meters.values()]
viz_mode = True

if __name__ == '__main__':
    idx = 0
    for image_path in image_paths_list:
        idx += 1
        image = cv2.imread(image_path)

        # Crop central ROI
        cropped_image, upper_left, _ = cv_utils.crop_region(image, x_center=image.shape[1] / 2, y_center=image.shape[0] / 2, x_pixels=2700, y_pixels=1700) # TODO: change to square
        if viz_mode:
            viz_utils.show_image('cropped image', cropped_image)

        # Estimate orchard orientation
        angle = trunks_detection.estimate_rows_orientation(cropped_image)
        if viz_mode:
            rotation_mat = cv2.getRotationMatrix2D((cropped_image.shape[1] / 2, cropped_image.shape[0] / 2), angle, scale=1.0) # TODO: center point ? (+ coordinates order)
            vertical_rows_image = cv2.warpAffine(cropped_image, rotation_mat, (cropped_image.shape[1], cropped_image.shape[0]))
            viz_utils.show_image('vertical rows', vertical_rows_image)

        # Get tree centroids
        centroids, rotated_centroids = trunks_detection.find_tree_centroids(cropped_image, angle)
        if viz_mode:
            vertical_rows_centroids_image = cv_utils.draw_points_on_image(vertical_rows_image, itertools.chain.from_iterable(rotated_centroids), color=(0, 0, 255))
            viz_utils.show_image('vertical rows centroids', vertical_rows_centroids_image)
            # centroids_image = cv_utils.draw_points_on_image(cropped_image, centroids, color=(0, 0, 255))
            # viz_utils.show_image('centroids', centroids_image)

        # Estimate grid parameters
        delta_x, delta_y = trunks_detection.estimate_grid_dimensions(rotated_centroids)
        shear, drift_vectors = trunks_detection.estimate_shear(rotated_centroids)
        if viz_mode:
            drift_vectors_image = cv_utils.draw_lines_on_image(vertical_rows_centroids_image, drift_vectors, color=(255, 255, 0))
            viz_utils.show_image('drift vectors', drift_vectors_image)

        # Get essential grid
        essential_grid = trunks_detection.get_essential_grid(delta_x, delta_y, shear, angle * (-1), n=4) # TODO: angle * (-1) ?
        if viz_mode:
            essential_grid_shape = np.max(essential_grid, axis=0) - np.min(essential_grid, axis=0)
            margin = essential_grid_shape * 0.2
            essential_grid_shifted = [tuple(elem) for elem in np.array(essential_grid) - np.min(essential_grid, axis=0) + margin / 2]
            estimated_grid_image = np.full((int(essential_grid_shape[1] + margin[1]), int(essential_grid_shape[0] + margin[0]), 3), 0, dtype=np.uint8)
            estimated_grid_image = cv_utils.draw_points_on_image(estimated_grid_image, essential_grid_shifted, color=(255, 0, 0))
            viz_utils.show_image('estimated grid', estimated_grid_image)

        # Find an origin to the grid
        positioned_grid, origin, drift_vectors = trunks_detection.find_min_mse_position(centroids, essential_grid, cropped_image.shape[1], cropped_image.shape[0])
        if viz_mode:
            positioned_grid_image = cv_utils.draw_points_on_image(cropped_image, positioned_grid, color=(255, 0, 0), radius=20)
            positioned_grid_image = cv_utils.draw_points_on_image(positioned_grid_image, centroids, color=(0, 0, 255), radius=10)
            positioned_grid_image = cv_utils.draw_lines_on_image(positioned_grid_image, drift_vectors, color=(255, 255, 0), thickness=3)
            viz_utils.show_image('positioned grid', positioned_grid_image)


        # grid = trunks_detection.get_grid(delta_x, delta_y, origin, angle * (-1), shear, n=4)
        # if viz_mode:
        #     grid_image = cv_utils.draw_points_on_image(cropped_image, grid, color=(255, 0, 0), radius=20)
        #     viz_utils.show_image('sanity check grid', grid_image)

        # break