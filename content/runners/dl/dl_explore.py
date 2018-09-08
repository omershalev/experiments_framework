import cv2
import numpy as np
from itertools import product
import pickle
import rospy
import torch
import torch.utils.data as data
from joblib import Parallel, delayed

from experiments_framework.framework import ros_utils
from air_ground_orchard_navigation.computer_vision import segmentation, contours_scan2
from experiments_framework.content.data_pointers.lavi_april_18 import panorama
from experiments_framework.framework import cv_utils
from experiments_framework.framework import viz_utils
from experiments_framework.framework import ros_utils
from experiments_framework.framework import config

def iteration_handler(map_image, x, y):
    if map_image[(y, x)] != 0:
        return None
    print (x,y)
    laser_range, _ = contours_scan2.generate(map_image, center_x=x, center_y=y, min_angle=0,
                                             max_angle=2 * np.pi, samples_num=360, min_distance=3,
                                             max_distance=300, resolution=config.top_view_resolution)  # TODO: resolution
    return laser_range

def joblib_map(fn, *iterables):
    iterable = list(zip(*iterables))
    return Parallel(
        n_jobs=-1,
        batch_size='auto'
    )(delayed(fn)(*args) for args in iterable)

# class LaserScanDataset(data.Dataset):
#     def __init__(self, path_to_pickle):
#         with open(path_to_pickle, 'r') as p:
#             self.data_obj = pickle.load(p)
#
#     def __getitem__(self, index):
#         if type(index) is int:
#             torch.tensor()
#         else:
#             raise Exception('unsupported format')
#
#     def __len__(self):
#         return len(self.data_obj)


map_image = cv2.cvtColor(cv2.imread(r'/home/omer/Downloads/temp_map_2.pgm'), cv2.COLOR_RGB2GRAY)

def slice_handler((left_x, right_x)):
    dataset = {}
    total = (right_x - left_x) * map_image.shape[0]
    i = 0
    for x in xrange(left_x, right_x):
        for y in xrange(0, map_image.shape[0]):
            if map_image[(y, x)] == 0:
                # print (x,y)
                print (100 * float(i) / float(total))
                laser_range, _ = contours_scan2.generate(map_image, center_x=x, center_y=y, min_angle=0,
                                                         max_angle=2 * np.pi, samples_num=360,
                                                         min_distance=3, max_distance=300,
                                                         resolution=0.0125)  # TODO: resolution
                dataset[(x,y)] = laser_range
            i += 1
    return dataset

def omer(x):
    return x ** 2

if __name__ == '__main__':

    import datetime
    start = datetime.datetime.now()
    # xx = list(np.linspace(start=0, stop=map_image.shape[1], num=8))
    xx = list(np.arange(start=0, stop=map_image.shape[1], step=350))

    x_start_stop_tuples = [(x_start, x_stop) for x_start, x_stop in zip(xx, xx[1:])]
    sub_datasets = Parallel(n_jobs=-1, timeout=36000)(delayed(slice_handler)(x_tuple) for x_tuple in x_start_stop_tuples)
    # sub_datasets = Parallel(n_jobs=-1)(delayed(lambda x_start_stop_tuple: slice_handler(x_start_stop_tuple[0], 0, x_start_stop_tuple[1], map_image.shape[0]))(x_start_stop_tuple) for x_start_stop_tuple in x_start_stop_tuples)
    # sub_datasets = Parallel(n_jobs=-1)(delayed(lambda x_start_stop_tuple: slice_handler(x_start_stop_tuple[0], 0, x_start_stop_tuple[1], map_image.shape[0])) for x_start_stop_tuple in x_start_stop_tuples)
    # sub_datasets = Parallel(n_jobs=-1)(delayed(omer)(i) for i in range(10))
    end = datetime.datetime.now()
    delta = end-start
    with open(r'/home/omer/Downloads/scan_dataset_3.pkl', 'wb') as p:
        pickle.dump(sub_datasets, p)
    print (delta)
    print ('end')

    # image_path = panorama.full_orchard['dji_afternoon'].path
    # image = cv2.imread(image_path)
    # map_image = segmentation.extract_canopies_map(image)

    # map_image = cv2.cvtColor(cv2.imread(r'/home/omer/Downloads/temp_map_3.pgm'), cv2.COLOR_RGB2GRAY)
    # keys = product(xrange(map_image.shape[1]), xrange(map_image.shape[0]))
    # dataset = zip(keys, map(lambda (x, y): iteration_handler(map_image, x, y), keys))
    #
    # with open(r'/home/omer/Downloads/scan_dataset_2.pkl', 'wb') as p:
    #     pickle.dump(dataset, p)

    # map_image = cv2.cvtColor(cv2.imread(r'/home/omer/Downloads/temp_map_2.pgm'), cv2.COLOR_RGB2GRAY)
    # dataset = {}
    # for x, y in product(xrange(map_image.shape[1]), xrange(map_image.shape[0])):
    #     if map_image[(y, x)] == 0:
    #         print (x, y)
    #         # map_image = cv2.circle(map_image, (x, y), radius=10, color=(255, 255, 255), thickness=2)
    #         laser_range, coordinates_list = contours_scan.generate(map_image, center_x=x, center_y=y, min_angle=0, max_angle=2*np.pi, samples_num=360, min_distance=3, max_distance=300, resolution=0.0125) # TODO: resolution
    #         # laser_range = iteration_handler(map_image, x, y)
    #         dataset[(x,y)] = laser_range
    # with open(r'/home/omer/Downloads/scan_dataset_2.pkl', 'wb') as p:
    #     pickle.dump(dataset, p)
    # viz_utils.show_image('points', map_image)

