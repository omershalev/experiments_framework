import cv2
import numpy as np
from scipy.optimize import minimize, fmin, basinhopping
import concurrent.futures
from itertools import product

import experiments_framework.framework.viz_utils as viz_utils
import air_ground_orchard_navigation.computer_vision.image_alignment as image_alignment
import air_ground_orchard_navigation.computer_vision.segmentation as canopy_contours


def objective(transformation_args, baseline_img, target_image):
    dx = transformation_args[0]
    dy = transformation_args[1]
    dtheta = transformation_args[2]
    scale = transformation_args[3]
    # print (dx, dy, dtheta, scale)
    M_translation = np.float32([[1, 0, dx], [0, 1, dy]])
    rows, cols = target_image.shape
    M_rotation_and_scale = cv2.getRotationMatrix2D(center=(cols / 2, rows / 2), angle=dtheta, scale=scale)
    translated_img = cv2.warpAffine(target_image, M_translation, (cols, rows))
    transformed_img = cv2.warpAffine(translated_img, M_rotation_and_scale, (cols, rows))
    return np.sum(np.sum((transformed_img - baseline_img) ** 2))  # TODO: consider smart image diff


if __name__ == '__main__':

    import experiments_framework.content.data_pointers.lavi_april_18.dji as dji_data

    noon = dji_data.snapshots_80_meters['15-08-1']
    late_noon = dji_data.snapshots_80_meters['15-53-1']
    afternoon = dji_data.snapshots_80_meters['16-55-1']
    # late_afternoon = dji_data.snapshots_80_meters['19-04-1']

    img_noon = cv2.imread(noon.path)
    _, noon_canopy_contours = canopy_contours.extract_canopy_contours(img_noon)
    img_late_noon = cv2.imread(late_noon.path)
    _, late_noon_canopy_contours = canopy_contours.extract_canopy_contours(img_late_noon)
    img_afternoon = cv2.imread(afternoon.path)
    _, afternoon_canopy_contours = canopy_contours.extract_canopy_contours(img_afternoon)

    # opt = minimize(fun=objective, x0=np.array([0,0,0,1.1]), args=(noon_canopy_contours, afternoon_canopy_contours),
    #                method='Nelder-Mead',
    #                bounds=[(-500, 500), (-500, 500), (0, 360), (0.5, 1.5)])
    # print opt.x

    dx_init_values = np.linspace(start=-100, stop=100, num=2)
    dy_init_values = np.linspace(start=-100, stop=100, num=2)
    dtheta_init_values = np.linspace(start=-20, stop=20, num=2)
    scale_init_values = np.linspace(start=0.9, stop=1.1, num=2)
    init_values_combinations = list(product(dx_init_values, dy_init_values, dtheta_init_values, scale_init_values))

    def optimize(init_values):
        x_opt = fmin(func=objective, x0=np.array(init_values), args=(noon_canopy_contours, afternoon_canopy_contours),
                       xtol=0.5, ftol=0.5)
        print x_opt
        return objective(x_opt, noon_canopy_contours, afternoon_canopy_contours)


    init_values_to_objective = {}

    # for init_values, objective_value in zip(init_values_combinations, map(optimize, init_values_combinations)):
    #     init_values_to_objective[init_values] = objective_value

    # TODO: check the following
    # cv2.estimateAffine2D()
    # cv2.estimateAffinePartial2D()
    # cv2.estimateRigidTransform()

    with concurrent.futures.ProcessPoolExecutor(max_workers=4) as executor:
        for init_values, objective_value in zip(init_values_combinations, executor.map(optimize, init_values_combinations)):
            init_values_to_objective[init_values] = objective_value

    print init_values_to_objective

    # x_opt = fmin(func=objective, x0=np.array([0,0,0,1]), args=(noon_canopy_contours, afternoon_canopy_contours),
    #                xtol=0.5, ftol=0.5)

    # x_opt = basinhopping(func=lambda x: objective(x, noon_canopy_contours, afternoon_canopy_contours), x0=np.array([0,0,0,1]))
    # print x_opt

    # img_late_noon = cv2.imread(late_noon.path)
    # img_afternoon = cv2.imread(afternoon.path)
    # img_late_afternoon = cv2.imread(late_afternoon.path)
    #
    # viz_utils.show_image('noon', img_noon)
    # viz_utils.show_image('late noon', img_late_noon)
    # viz_utils.show_image('afternoon', img_afternoon)
    # viz_utils.show_image('late afternoon', img_late_afternoon)
    print ('hi')



