import os
import cv2
import json

from framework import cv_utils
from framework import utils


#################################################################################################
#                                             CONFIG                                            #
#################################################################################################
altitude = 80
repetitions = 1
grid_size_values = [6]

setup = 'apr' # apr / nov1 / nov2 / nov3 / nov4
#################################################################################################

if setup == 'apr':
    from content.data_pointers.lavi_april_18.dji import snapshots_80_meters as snapshots
    from content.data_pointers.lavi_april_18.orchard_topology import plot_pattern as plot_pattern
elif setup == 'nov1':
    from content.data_pointers.lavi_november_18.dji import plot1_snapshots_80_meters as snapshots
    from content.data_pointers.lavi_november_18.orchard_topology import plot1_pattern as plot_pattern
elif setup == 'nov2':
    from content.data_pointers.lavi_november_18.dji import plot2_snapshots_80_meters as snapshots
    from content.data_pointers.lavi_november_18.orchard_topology import plot2_pattern as plot_pattern
elif setup == 'nov3':
    from content.data_pointers.lavi_november_18.dji import plot3_snapshots_80_meters as snapshots
    from content.data_pointers.lavi_november_18.orchard_topology import plot3_pattern as plot_pattern
elif setup == 'nov4':
    from content.data_pointers.lavi_november_18.dji import plot4_snapshots_80_meters as snapshots
    from content.data_pointers.lavi_november_18.orchard_topology import plot4_pattern as plot_pattern


image_keys = ['16-55-1', '19-04-1']

if __name__ == '__main__':
    execution_dir = utils.create_new_execution_folder('trunks_tagging')
    for image_key in image_keys:
        summary = {}
        summary['results'] = {}
        summary['results'][0] = {}
        summary['results'][0]['semantic_trunks'] = {}
        summary['metadata'] = {}
        summary['metadata']['image_key'] = image_key
        summary['metadata']['altitude'] = altitude
        data_descriptor = snapshots[image_key]
        summary['data_sources'] = data_descriptor.path
        image = cv2.imread(data_descriptor.path)
        for i in range(plot_pattern.shape[0]):
            for j in range(plot_pattern.shape[1]):
                if plot_pattern[(i, j)] == -1:
                    continue
                tree_label = '%d/%s' % (j + 1, chr(65 + (plot_pattern.shape[0] - 1 - i)))
                trunk_pose = cv_utils.sample_pixel_coordinates(image)
                summary['results'][0]['semantic_trunks'][tree_label] = trunk_pose
                image = cv_utils.draw_points_on_image(image, [trunk_pose], color=(255, 255, 255))
                cv2.putText(image, tree_label, trunk_pose, fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                            fontScale=2, color=(255, 255, 255), thickness=8, lineType=cv2.LINE_AA)
        # typical values
        summary['results'][0]['optimized_grid_dim_x'] = 270
        summary['results'][0]['optimized_grid_dim_y'] = 230
        summary['results'][0]['optimized_sigma'] = 80
        with open(os.path.join(execution_dir, '%s_trunks.json' % image_key), 'w') as f:
            json.dump(summary, f, indent=4)
