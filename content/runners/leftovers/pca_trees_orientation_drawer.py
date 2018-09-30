import os
import cv2
import numpy as np

import computer_vision.segmentation as canopy_contours
from framework import viz_utils
from framework import cv_utils
import content.data_pointers.lavi_april_18.dji as dji_data

image_paths_list = [descriptor.path for descriptor in dji_data.snapshots_60_meters.values() + dji_data.snapshots_80_meters.values()]
show = True


def get_orientation(points, image):
    mean, eigenvectors = cv2.PCACompute(points, mean=np.empty(0))
    center_of_mass = (int(mean[0, 0]), int(mean[0, 1]))
    # p1 = (center_of_mass[0] + eigenvectors[0, 0] * 1000, center_of_mass[1] + eigenvectors[0, 1] * 1000)
    p1 = (center_of_mass[0] + eigenvectors[0, 1] * 1000, center_of_mass[1] + eigenvectors[0, 0] * 1000) # TODO: is this the correct order?
    # p2 = (center_of_mass[0] + eigenvectors[1, 0] * 500, center_of_mass[1] + eigenvectors[1, 1] * 500)
    p2 = (center_of_mass[0] + eigenvectors[1, 1] * 500, center_of_mass[1] + eigenvectors[1, 0] * 500) # TODO: is this the correct order?
    image = cv2.line(image, (int(center_of_mass[0]), int(center_of_mass[1])), (int(p1[0]), int(p1[1])), (255, 255, 0), 7, cv2.LINE_AA)
    image = cv2.line(image, (int(center_of_mass[0]), int(center_of_mass[1])), (int(p2[0]), int(p2[1])), (255, 255, 0), 7, cv2.LINE_AA)
    return image

if __name__ == '__main__':
    idx = 0
    for image_path in image_paths_list:
        image = cv2.imread(image_path)
        # cropped_image = cv_utils.center_crop(image, 0.25, 0.25)
        cropped_image = cv_utils.crop_region(image, x_center=image.shape[1]/2, y_center=image.shape[0]/2, x_pixels=2700, y_pixels=1700)
        contours, contours_mask = canopy_contours.extract_canopy_contours(cropped_image)
        cv2.drawContours(cropped_image, contours, contourIdx=-1, color=(0, 255, 0), thickness=3)
        idx += 1

        all_contours_points = np.concatenate([contour for contour in contours])
        all_contours_points_2d_array = np.empty((len(all_contours_points), 2), dtype=np.float64)
        for i in range(all_contours_points_2d_array.shape[0]):
                all_contours_points_2d_array[i, 0] = all_contours_points[i, 0, 0]
                all_contours_points_2d_array[i, 1] = all_contours_points[i, 0, 1]

        cropped_image = get_orientation(all_contours_points_2d_array, cropped_image)

        if show:
            viz_utils.show_image('image', cropped_image)
            # viz_utils.show_image('image', image)


        # TODO: try all the friends below on the data extracted from ORB/SIFT but on the contours mask
        # cv2.estimateAffine2D()
        # cv2.estimateAffinePartial2D()
        # cv2.estimateRigidTransform()

        # break