import cv2
import numpy as np

def generate(map_image, center_x, center_y, min_angle, max_angle, samples_num, min_distance, max_distance, resolution):
    scan_ranges = np.full(samples_num, np.nan)
    scan_coordinates_list = []
    scan_index = 0
    for theta in np.linspace(min_angle, max_angle, num=samples_num):
        for r in np.linspace(min_distance, max_distance):
    # for theta in np.linspace(start=min_angle, stop=max_angle, num=samples_num):
    #     for r in np.arange(start=min_distance, stop=max_distance, step=1):
            x = int(np.round(center_x + r * np.cos(-theta)))
            y = int(np.round(center_y + r * np.sin(-theta)))
            if not (0 <= x < np.size(map_image, 1) and 0 <= y < np.size(map_image, 0)):
                break
            if map_image[y, x] in [255, 128]: # TODO: verify order
                scan_ranges[scan_index] = (np.sqrt((center_x - x) ** 2 + (center_y - y) ** 2)) * resolution
                scan_coordinates_list.append((x,y))
                break
            # if map_image[y, x] != 0: # TODO: is that condition needed?
            #     break
        scan_index += 1
    return scan_ranges, scan_coordinates_list