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
    for image_path in image_paths_list:

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

        '''
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



        import pickle
        obj_to_dump = {}
        obj_to_dump['optimized_grid'] = optimized_grid
        obj_to_dump['optimized_grid_args'] = optimized_grid_args
        with open(r'/home/omer/Downloads/obj_to_dump.pkl', 'wb') as p:
            pickle.dump(obj_to_dump, p)

        break
        '''


        # Extrapolate rest of the grid
        # extrapolated_grid = {}
        # for i in range(N):
        #     for j in range(N):
        #         extrapolated_grid[(i, j)] = np.array(optimized_grid[i + N * j]) + np.array(crop_origin)



        # DATA LOAD #
        import pickle
        with open(r'/home/omer/Downloads/obj_to_dump.pkl') as p:
            obj_to_dump = pickle.load(p)
        optimized_grid = obj_to_dump['optimized_grid']
        optimized_grid_args = obj_to_dump['optimized_grid_args']
        optimized_grid_dim_x, optimized_grid_dim_y, optimized_translation_x, optimized_translation_y, optimized_orientation, optimized_shear, optimized_sigma = optimized_grid_args
        # E/O DATA LOAD #



        alpha = np.arcsin(optimized_shear)


        # full_grid_optimized_translation_x = optimized_translation_x + np.array(crop_origin)[0]
        # full_grid_optimized_translation_y = optimized_translation_y + np.array(crop_origin)[1]
        #
        # vertical_delta = optimized_grid_dim_x * np.cos(alpha)
        # horizontal_delta = optimized_grid_dim_y * np.sin(alpha)


        ####################
        optimized_grid_shifted_origin = np.array(optimized_grid[0]) + np.array(crop_origin)
        ####################

        optimized_grid_shifted = np.array(optimized_grid) + np.array(crop_origin)

        full_grid = trunks_detection.get_essential_grid(optimized_grid_dim_x, optimized_grid_dim_y, optimized_shear, optimized_orientation, n=3 * N)
        full_grid = np.array(full_grid) - np.array(full_grid[0]) + optimized_grid_shifted_origin # TODO: KEEPPPPPPPPPPPPPPPP

        rotation_pivot = (full_grid[0][0], full_grid[0][1])

        rotation_mat = np.insert(cv2.getRotationMatrix2D(rotation_pivot, (-1) * optimized_orientation, scale=1.0), [2], [0, 0, 1], axis=0) # TODO: * (-1) ?
        full_grid_np = full_grid.reshape(-1, 1, 2)
        rotated_full_grid_np = cv2.perspectiveTransform(full_grid_np, rotation_mat)

        rotated_full_grid = [tuple(elem) for elem in rotated_full_grid_np[:, 0, :].tolist()]
        rotated_full_grid_np = (np.array(rotated_full_grid) - np.array([4 * optimized_grid_dim_x, 4 * (optimized_grid_dim_x * np.tan(alpha) + optimized_grid_dim_y)])).reshape(-1, 1, 2)

        rotation_mat = np.insert(cv2.getRotationMatrix2D(rotation_pivot, optimized_orientation, scale=1.0), [2], [0, 0, 1], axis=0) # TODO: * (-1) ?
        full_grid_np = cv2.perspectiveTransform(rotated_full_grid_np, rotation_mat)
        full_grid = [tuple(elem) for elem in full_grid_np[:, 0, :].tolist()]

        # full_grid = np.array(full_grid) - np.array(full_grid[0]) # TODO: KEEPPPPPPPPPPPPPPPP
        # full_grid = np.array(full_grid) - np.array(full_grid[0]) + np.array([-50, -50]) # TODO: KEEPPPPPPPPPPPPPPPP

        stam = cv_utils.draw_points_on_image(image, full_grid, color=(0, 0, 255))
        stam = cv_utils.draw_points_on_image(stam, optimized_grid_shifted, color=(255, 0, 0))
        viz_utils.show_image('stam', stam)

        # TODO: you must filter out points with negative coordinates
        break
        # interpolated_grid_full_image = {}
        # for key in ordered_grid:
        #     ordered_grid[key] = np.array(ordered_grid[key]) + np.array(crop_origin)
        # optimized_grid_image = cv_utils.draw_points_on_image(image, ordered_grid.values(), color=(255, 0, 0))


        # horizontal_delta = 1.0 * (ordered_grid[(N - 1, 0)][1] - ordered_grid[(0, 0)][1]) / N
        # vertical_delta = 1.0 * (ordered_grid[(0, N - 1)][0] - ordered_grid[(0, 0)][0]) / N
        # full_grid_origin_x = ordered_grid[(0, 0)][0] % vertical_delta
        # full_grid_origin_y = ordered_grid[(0, 0)][1] % horizontal_delta


        # diff_vector = ordered_grid[(0, 0)] - ordered_grid[(0, N - 1)]
        # alpha = np.arctan2(diff_vector[1], diff_vector[0]) # TODO: can use shear for that!
        # d = np.sqrt((upper_right[0] - upper_left[0]) ** 2 + (upper_right[1] - upper_left[1]) ** 2) / N


        # full_grid_origin_x = ordered_grid[(0, 0)][1] - horizontal_delta
        # full_grid_translation_x = ordered_grid[(0, 0)][0]
        # full_grid_origin_y = ordered_grid[(0, 0)][0] - vertical_delta
        # full_grid_translation_y = ordered_grid[(0, 0)][1]

        full_grid = trunks_detection.get_grid(optimized_grid_dim_x, optimized_grid_dim_y,
                                              (optimized_translation_x + crop_origin[0], optimized_translation_y + crop_origin[1]), optimized_orientation, optimized_shear, n=6)
        # upper_left = np.array(interpolated_grid[(0, 0)])
        # upper_right = np.array(interpolated_grid[(0, N - 1)])


        # new_point = np.array([upper_right[0] + d * np.cos(alpha), upper_right[1] + d * np.sin(alpha)])
        # points_to_draw = [tuple(new_point), tuple(upper_right), tuple(upper_left)]
        stam = cv_utils.draw_points_on_image(image, full_grid, color=(0, 0, 255))
        # stam = cv_utils.draw_points_on_image(stam, ordered_grid.values(), color=(255, 0, 0))
        stam = cv_utils.draw_points_on_image(stam, [ordered_grid[(0,0)], ordered_grid[0,5]], color=(255, 0, 0))
        # stam = cv_utils.draw_points_on_image(image, [(full_grid_origin_x, full_grid_origin_y)], color=(0, 0, 255))
        viz_utils.show_image('stam', stam)
        print ('end of iteration')
        break