import os
import cv2
import numpy as np
from itertools import product

from computer_vision import segmentation
from framework import viz_utils
from framework import cv_utils
from framework import utils
from content.data_pointers.lavi_april_18 import dji

image_paths_list = [descriptor.path for descriptor in dji.snapshots_60_meters.values() + dji.snapshots_80_meters.values()]
show = True


# def get_sqare_gaussian_image(sigma, size):
#     x, y = np.meshgrid(np.linspace(-1, 1, size), np.linspace(-1, 1, size))
#     d = np.sqrt(x * x + y * y)
#     return np.exp(-((d) ** 2 / (2.0 * sigma ** 2)))

def get_gaussian_on_image(mu_x, mu_y, sigma, size_x, size_y):
    image = np.full((size_y, size_x), fill_value=0, dtype=np.float64)
    x_start, x_end = max(0, int(mu_x - 3 * sigma)), min(size_x, int(mu_x + 3 * sigma))
    y_start, y_end = max(0, int(mu_y - 3 * sigma)), min(size_y, int(mu_y + 3 * sigma))
    x, y = np.meshgrid(np.arange(x_start, x_end), np.arange(y_start, y_end))
    squre_gaussian = np.exp(-((x - mu_x) ** 2 + (y - mu_y) ** 2) / (2.0 * sigma ** 2))
    image = cv_utils.insert_image_patch(image, squre_gaussian,
                                        upper_left=(x_start, y_start),
                                        lower_right=(x_end, y_end))
    return image


def get_basic_grid(upper_left_node, lower_right_node, nodes_num_x, nodes_num_y):
    centroids = list(product(np.linspace(upper_left_node[0], lower_right_node[0], num=nodes_num_x),
                             np.linspace(upper_left_node[1], lower_right_node[1], num=nodes_num_y)))
    return centroids


def get_transformed_grid(basic_grid, delta_x, delta_y, angle, scale):
    basic_grid_np = np.float32(basic_grid).reshape(-1, 1, 2)
    translation_mat = np.float32([[1, 0, delta_x], [0, 1, delta_y], [0, 0, 1]])
    transformed_grid_np = cv2.perspectiveTransform(basic_grid_np, translation_mat)
    rotation_scale_mat = np.insert(cv2.getRotationMatrix2D((basic_grid[0][0], basic_grid[0][1]), angle, scale), [2], [0, 0, 1], axis=0)
    transformed_grid_np = cv2.perspectiveTransform(transformed_grid_np, rotation_scale_mat)
    transformed_grid = [tuple(elem) for elem in transformed_grid_np[:,0,:].tolist()]
    return transformed_grid
    # TODO: does the order matter? I first translate and then rotate + scale
    # TODO: did I build the two transformations correctly?
    # TODO: order of cos/sin in https://docs.opencv.org/2.4/modules/imgproc/doc/geometric_transformations.html is different than in https://en.wikipedia.org/wiki/Rotation_matrix
    # TODO: why does perspectiveTransform require 3x3 matrix? in homography transformation, the H is passed




if __name__ == '__main__':
    idx = 0

    for image_path in image_paths_list:
        image = cv2.imread(image_path)
        cropped_image, _, _ = cv_utils.crop_region(image, x_center=image.shape[1]/2, y_center=image.shape[0]/2, x_pixels=2700, y_pixels=1700)
        # points = cv_utils.sample_pixel_coordinates(cropped_image, multiple=True)
        points = [(541, 403), (2281, 1263)]

        basic_grid = get_basic_grid(points[0], points[1], nodes_num_x=6, nodes_num_y=4)

        transformed_grid = get_transformed_grid(basic_grid, delta_x=0, delta_y=0, angle=30, scale=1.0)


        gaussians = np.full((np.size(cropped_image, 0), np.size(cropped_image, 1)), fill_value=0, dtype=np.float64)
        for x, y in transformed_grid:
            gaussian_sigma = 70
            if x < 0 or y < 0:
                continue
            gaussian = get_gaussian_on_image(x, y, gaussian_sigma, cropped_image.shape[1], cropped_image.shape[0])
            gaussians = np.add(gaussians, gaussian)
        viz_utils.show_image('gaussians', gaussians)
        _, contours_mask = segmentation.extract_canopy_contours(cropped_image)
        mult_res = np.multiply(contours_mask, gaussians)



        # for x, y in transformed_grid:
        #     cv2.circle(cropped_image, (int(x), int(y)), radius=15, color=(0, 0, 255), thickness=-1)
        # viz_utils.show_image('p', cropped_image)

        # g = get_sqare_gaussian_image(0.15, 300)
        # gg = np.tile(g, (5, 7)) # TODO: the problem with tile is that it doesn't allow overlaps - maybe that's okay?
        # gg = np.pad(gg, ((100, 100), (300, 300)), 'constant', constant_values=(0))
        # cropped_image = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2GRAY)

        # M_rotation_and_scale = cv2.getRotationMatrix2D(center=(cols / 2, rows / 2), angle=15, scale=0.8)
        # gg = cv2.warpAffine(gg, M_translation, (cols, rows))
        # gg = cv2.warpAffine(gg, M_rotation_and_scale, (cols, rows))

        # viz_utils.show_image('gaussian', gg)
        # viz_utils.show_image('contours', contours_mask)
        # viz_utils.show_image('masked', masked)

        idx += 1

        break