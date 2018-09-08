import cv2
import numpy as np
import pickle
import multiprocessing

from air_ground_orchard_navigation.computer_vision.contours_scan_cython import contours_scan
from air_ground_orchard_navigation.computer_vision import contours_scan2
from experiments_framework.framework import utils


def slice_handler(map_image, left_x, right_x, upper_y, lower_y):
    total = (right_x - left_x) * (lower_y - upper_y)
    i = 0
    for x in xrange(left_x, right_x):
        for y in xrange(upper_y, lower_y):
            if map_image[(y, x)] == 0:
                print (100 * float(i) / float(total))
                laser_range, _ = contours_scan2.generate(map_image, center_x=x, center_y=y, min_angle=0, # TODO: decide what to do with the non-cython version (need to make it compliant with the other one)
                                                         max_angle=2 * np.pi, samples_num=360,
                                                         min_distance=3, max_distance=300,
                                                         resolution=0.0125)  # TODO: resolution
                with open(r'/home/omer/Downloads/temp/%d-%d.pkl' % (x, y), 'wb') as p:
                    pickle.dump(laser_range, p)
            i += 1

if __name__ == '__main__':

    import datetime

    map_image = cv2.cvtColor(cv2.imread(r'/home/omer/Downloads/temp_map_2.pgm'), cv2.COLOR_RGB2GRAY)
    slice_step = map_image.shape[1] / multiprocessing.cpu_count()
    x_splits = range(0, map_image.shape[1], slice_step)
    x_splits[-1] = map_image.shape[1]
    x_start_stop_tuples = [(x_start, x_stop) for x_start, x_stop in zip(x_splits, x_splits[1:])]
    utils.joblib_map(slice_handler, [(map_image, x_start, x_stop, 0, map_image.shape[0]) for x_start, x_stop in x_start_stop_tuples])
    # utils.joblib_map(slice_handler, [(map_image, x_start, x_stop, 0, 5) for x_start, x_stop in x_start_stop_tuples])

    print (datetime.datetime.now())
    print ('end')

