import os
import cv2
import json

from framework import cv_utils
from framework import viz_utils
from framework import config
# from content.data_pointers.lavi_april_18 import dji
from content.data_pointers.lavi_november_18 import dji

# data_dicts_list = {'plot3_snapshots_60_meters': dji.plot3_snapshots_60_meters, 'plot3_snapshots_80_meters': dji.plot3_snapshots_80_meters}
data_dicts_list = {'plot4_snapshots_80_meters': dji.plot4_snapshots_80_meters}

if __name__ == '__main__':
    for data_dict_name, data_dict in data_dicts_list.items():
        marker_positions = {}
        for key, data_descriptor in data_dict.items():
            image = cv2.imread(data_descriptor.path)
            print ('click on upper left marker')
            x_ul, y_ul = cv_utils.sample_pixel_coordinates(image)
            print ('click on upper right marker')
            x_ur, y_ur = cv_utils.sample_pixel_coordinates(image)
            print ('click on lower right marker')
            x_lr, y_lr = cv_utils.sample_pixel_coordinates(image)
            print ('click on lower left marker')
            x_ll, y_ll = cv_utils.sample_pixel_coordinates(image)
            points = [(x_ul, y_ul), (x_ur, y_ur), (x_lr, y_lr), (x_ll, y_ll)]
            image = cv_utils.mark_rectangle_on_image(image, points)
            viz_utils.show_image('rectangle', image)

            marker_positions[key] = [(x_ul, y_ul), (x_ur, y_ur), (x_lr, y_lr), (x_ll, y_ll)]

        with open(os.path.join(config.temp_output_path, data_dict_name + '.json'), 'w') as j:
            json.dump(marker_positions, j, indent=4)

