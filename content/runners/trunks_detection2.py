import os
import json

from framework import utils
from content.experiments.trunks_detection2 import TrunksDetectionExperiment

#################################################################################################
#                                             CONFIG                                            #
#################################################################################################
description = 'trunks_detection'
altitude = 80
repetitions = 1
grid_size_values = [6]
setup = 'nov2' # apr / nov1 / nov2 / nov3 / nov4
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


if __name__ == '__main__':
    execution_dir = utils.create_new_execution_folder(description)
    for image_key in snapshots.keys():

        # goods: 11-07-18
        # if image_key in ['11-07-18', '11-07-19', '11-07-16', '11-07-17']: # TODO: remove
        #     continue

        image_descriptor = snapshots[image_key]

        for grid_size in grid_size_values:

            # Run experiment
            experiment = TrunksDetectionExperiment(name='trunks_detection_on_%s' % image_key, data_sources=image_descriptor.path, working_dir=execution_dir,
                                                   params={'crop_ratio': 0.8, 'initial_sigma_to_dim_y_ratio': 0.33, 'grid_size_for_optimization': grid_size,
                                                           'orchard_pattern': plot_pattern}, metadata={'image_key': image_key, 'altitude': altitude})
            experiment.run(repetitions, viz_mode=False)