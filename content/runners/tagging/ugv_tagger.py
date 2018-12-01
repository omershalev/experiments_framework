import os
import cv2
import json

from framework import cv_utils
from framework import utils
from content.data_pointers.lavi_november_18.dji import plot1_snapshots_80_meters as snapshots
from framework import logger
from collections import OrderedDict

_logger = logger.get_logger()

relevant_image_keys = ['10-26-3', '10-27-1', '10-27-2', '10-28-1', '10-29-1', '10-29-2', '10-30-1', '10-30-2', '10-31-1', '10-32-1']

if __name__ == '__main__':

    execution_dir = utils.create_new_execution_folder('ugv_tagging')
    ugv_poses = OrderedDict()
    for image_key in relevant_image_keys:
        image = cv2.imread(snapshots[image_key].path)
        height = image.shape[0]
        ugv_pose = list(cv_utils.sample_pixel_coordinates(image))
        ugv_pose[1] = height - ugv_pose[1]
        ugv_poses[image_key] = tuple(ugv_pose)
    with open(os.path.join(execution_dir, 'ugv_poses.json'), 'w') as f:
        json.dump(ugv_poses, f, indent=4)
