import os
import numpy as np

############################################# PATHS #############################################
root_dir_path = r'/home/omer/orchards_ws'
base_raw_data_path = os.path.join(root_dir_path, 'data/lavi_apr_18/raw')
panorama_path = os.path.join(root_dir_path, 'data/lavi_apr_18/panorama')
markers_locations_path = os.path.join(root_dir_path, 'data/lavi_apr_18/markers_locations')
base_output_path = os.path.join(root_dir_path, 'output')

output_to_console = True

screen_resolution = (1920, 1080)

##################################### BEST KNOWN PARAMETERS #####################################
markers_bounding_box_expand_ratio = 0.15
synthetic_scan_target_frequency = 30
top_view_resolution = 0.0125
min_scan_angle = 0
max_scan_angle = 2 * np.pi
scan_samples_num = 360
min_scan_distance = 3
max_scan_distance = 300