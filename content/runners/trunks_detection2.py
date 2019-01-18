from framework import utils
from content.experiments.trunks_detection2 import TrunksDetectionExperiment

#################################################################################################
#                                             CONFIG                                            #
#################################################################################################
description = 'trunks_detection'
altitude = 80
repetitions = 1
grid_size_for_optimization = 6
crop_ratio = 0.8
initial_sigma_to_dim_y_ratio = 0.33
viz_mode = False
verbose_optimization = True
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


if __name__ == '__main__':
    execution_dir = utils.create_new_execution_folder(description)
    for image_key in snapshots.keys():
        image_descriptor = snapshots[image_key]
        experiment = TrunksDetectionExperiment(name='trunks_detection_on_%s' % image_key, data_sources=image_descriptor.path, working_dir=execution_dir,
                                               params={'crop_ratio': crop_ratio, 'initial_sigma_to_dim_y_ratio': initial_sigma_to_dim_y_ratio,
                                                       'grid_size_for_optimization': grid_size_for_optimization, 'orchard_pattern': plot_pattern},
                                               metadata={'image_key': image_key, 'altitude': altitude})
        experiment.run(repetitions, viz_mode=viz_mode, verbose_optimization=verbose_optimization)