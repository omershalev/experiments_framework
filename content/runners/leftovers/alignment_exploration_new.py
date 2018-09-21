import cv2
import numpy as np
import json
from scipy.optimize import minimize, fmin, basinhopping
import concurrent.futures
from itertools import product

from experiments_framework.framework import viz_utils
from experiments_framework.framework import cv_utils
from experiments_framework.framework import utils
import air_ground_orchard_navigation.computer_vision.image_alignment as image_alignment
import air_ground_orchard_navigation.computer_vision.segmentation as canopy_contours


def objective(baseline_img, target_img, transformation_args):
    dx = transformation_args[0]
    dy = transformation_args[1]
    dtheta = transformation_args[2]
    scale = transformation_args[3]
    M_translation = np.float32([[1, 0, dx], [0, 1, dy]])
    rows, cols = target_img.shape
    M_rotation_and_scale = cv2.getRotationMatrix2D(center=(cols / 2, rows / 2), angle=dtheta, scale=scale)
    translated_img = cv2.warpAffine(target_img, M_translation, (cols, rows))
    transformed_img = cv2.warpAffine(translated_img, M_rotation_and_scale, (cols, rows))
    # TODO: is the order correct? ask Amir!
    ret = np.sum(np.sum((transformed_img - baseline_img) ** 2))
    print str((dx, dy, dtheta, scale)) + ':' + str(ret)
    return np.sum(np.sum((transformed_img - baseline_img) ** 2))  # TODO: consider smart image diff


def mse(imageA, imageB):
    if imageA.shape != imageB.shape:
        raise Exception('Two images must have the same dimension')
    err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
    err /= float(imageA.shape[0] * imageA.shape[1])
    return err


from content.data_pointers.lavi_april_18 import dji
keys_60 = ['15-20-1']
keys_80 = ['15-08-1', '15-53-1', '16-55-1', '19-04-1']
markers_locations_json_path = dji.snapshots_80_meters_markers_locations_json_path



if __name__ == '__main__':


    img1 = cv2.imread(dji.snapshots_80_meters[keys_80[0]].path)
    img2 = cv2.imread(dji.snapshots_80_meters[keys_80[1]].path)
    cv2.imwrite(r'/home/omer/Downloads/align/image.jpg', img1)
    cv2.imwrite(r'/home/omer/Downloads/align/baseline_image.jpg', img2)
    contours_mask1 = cv_utils.center_crop(canopy_contours.extract_canopy_contours(img1)[1], 0.3, 0.3)
    contours_mask2 = cv_utils.center_crop(canopy_contours.extract_canopy_contours(img2)[1], 0.3, 0.3)
    # criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 5000, 1e-10)
    # _, MM = cv2.findTransformECC(contours_mask1, contours_mask2, np.eye(2, 3, dtype=np.float32), cv2.MOTION_AFFINE, criteria)
    # registered_img2_no_point = cv2.warpAffine(img2, MM, (img2.shape[1], img2.shape[0]))
    # cv2.imwrite(r'/home/omer/Downloads/align/img2_registered_no_points.jpg', registered_img2_no_point)


    with open(markers_locations_json_path) as f:
        markers_locations = json.load(f)
    points1 = markers_locations[keys_80[0]]
    points2 = markers_locations[keys_80[1]]


    homogaphic_registration = cv_utils.warp_image(img1, points1, points2, method='homographic')
    affine_registration = cv_utils.warp_image(img1, points1, points2, method='affine')
    rigid_registration = cv_utils.warp_image(img1, points1, points2, method='rigid')

    cv2.imwrite(r'/home/omer/Downloads/align/homogaphic_registration.jpg', homogaphic_registration)
    cv2.imwrite(r'/home/omer/Downloads/align/affine_registration.jpg', affine_registration)
    cv2.imwrite(r'/home/omer/Downloads/align/rigid_registration.jpg', rigid_registration)


    # _, contours_mask_registered1 = canopy_contours.extract_canopy_contours(registered_img1)
    # _, contours_mask_registered2 = canopy_contours.extract_canopy_contours(img2)
    #
    #
    # print cv_utils.calculate_image_diff(contours_mask_registered1, contours_mask_registered2, method='mse')
    # print cv_utils.calculate_image_diff(contours_mask_registered1, contours_mask_registered2, method='ssim')



    print ('end')
