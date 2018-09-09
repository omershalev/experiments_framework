import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.signal import argrelmax

import air_ground_orchard_navigation.computer_vision.segmentation as canopy_contours
from experiments_framework.framework import cv_utils
from experiments_framework.framework import viz_utils
import experiments_framework.content.data_pointers.lavi_april_18.dji as dji_data

image_paths_list = [descriptor.path for descriptor in dji_data.snapshots_60_meters.values() + dji_data.snapshots_80_meters.values()]
show = True

if __name__ == '__main__':
    for image_path in image_paths_list:
        points = []
        image = cv2.imread(image_path)
        cropped_image = cv_utils.crop_region(image, x_center=image.shape[1]/2, y_center=image.shape[0]/2, x_pixels=2700, y_pixels=1700)
        # viz_utils.show_image('stam', cropped_image)
        contours, contours_mask = canopy_contours.extract_canopy_contours(cropped_image)
        sums_vector = np.sum(contours_mask, axis=0) * (-1)
        rows_centers = find_peaks(sums_vector, distance=200, width=50)[0]
        for left_limit, right_limit in zip(rows_centers[:-1], rows_centers[1:]):
            slice = contours_mask[:,left_limit:right_limit]
            sums_vector_2 = np.sum(slice, axis=1)
            # plt.plot(sums_vector_2)
            # plt.show()
            trees_centers = find_peaks(sums_vector_2, distance=160, width=30)[0]
            print(trees_centers)
            # viz_utils.show_image('slice', slice)
            for tree_center in trees_centers:
                points.append((int((left_limit + right_limit)/2), tree_center))

        for point in points:
            cv2.circle(cropped_image, point, radius=15, color=(0,0,255), thickness=-1)
        viz_utils.show_image('trunks', cropped_image)
        print ('end of iteration')