import cv2
import numpy as np
from scipy.optimize import minimize, fmin, basinhopping
import concurrent.futures
from itertools import product

import experiments_framework.framework.viz_utils as viz_utils
from experiments_framework.framework import utils
import air_ground_orchard_navigation.computer_vision.image_alignment as image_alignment
import air_ground_orchard_navigation.computer_vision.segmentation as canopy_contours


def objective(transformation_args):
    baseline_img = noon_canopy_contours
    target_image = late_noon_canopy_contours
    dx = transformation_args[0]
    dy = transformation_args[1]
    dtheta = transformation_args[2]
    scale = transformation_args[3]
    M_translation = np.float32([[1, 0, dx], [0, 1, dy]])
    rows, cols = target_image.shape
    M_rotation_and_scale = cv2.getRotationMatrix2D(center=(cols / 2, rows / 2), angle=dtheta, scale=scale)
    translated_img = cv2.warpAffine(target_image, M_translation, (cols, rows))
    transformed_img = cv2.warpAffine(translated_img, M_rotation_and_scale, (cols, rows))
    ret = np.sum(np.sum((transformed_img - baseline_img) ** 2))
    print str((dx, dy, dtheta, scale)) + ':' + str(ret)
    return np.sum(np.sum((transformed_img - baseline_img) ** 2))  # TODO: consider smart image diff



def slice_handler(start_idx, stop_idx, init_values_combinations):
    print ('hi')
    # for init_values in slice:
    #     v = objective(init_values)

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


    # dx_init_values = np.linspace(start=-100, stop=100, num=2)
    # dy_init_values = np.linspace(start=-100, stop=100, num=2)
    # dtheta_init_values = np.linspace(start=-20, stop=20, num=2)
    # scale_init_values = np.linspace(start=0.9, stop=1.1, num=2)

    dx_init_values = set([-int(1.8 ** i) for i in range(4)] + [int(1.8 ** i) for i in range(4)])
    dy_init_values = set([-int(1.8 ** i) for i in range(4)] + [int(1.8 ** i) for i in range(4)])
    dtheta_init_values = set([-int(1.4 ** i) for i in range(4)] + [int(1.4 ** i) for i in range(4)]) # in degrees
    scale_init_values = np.linspace(start=0.8, stop=1.2, num=8)
    init_values_combinations = list(product(dx_init_values, dy_init_values, dtheta_init_values, scale_init_values))

    # for xx in init_values_combinations:
    #     objective(xx)
    # def optimize(init_values):
    #     x_opt = fmin(func=objective, x0=np.array(init_values), args=(noon_canopy_contours, afternoon_canopy_contours),
    #                    xtol=0.5, ftol=0.5)
    #     print x_opt
    #     return objective(x_opt, noon_canopy_contours, afternoon_canopy_contours)

    # import multiprocessing
    # slice_step = len(init_values_combinations) / multiprocessing.cpu_count()
    # x_splits = range(0, len(init_values_combinations), slice_step)
    # x_splits[-1] = len(init_values_combinations)
    # x_start_stop_tuples = [(x_start, x_stop) for x_start, x_stop in zip(x_splits, x_splits[1:])]


    # def omer(x):
    #     return x[0]
    y = utils.distribute_evenly_on_all_cores(objective, init_values_combinations)
    # utils.joblib_map(slice_handler, [(x_start, x_stop, init_values_combinations) for x_start, x_stop in x_start_stop_tuples])

    print ('end')
