import os

from framework import utils
from framework import config
from content.experiments.gmapping import Gmapping

if __name__ == '__main__':

    execution_dir = utils.create_new_execution_folder('gmapping')

    config.base_raw_data_path = os.path.join(config.root_dir_path, 'resources/lavi_nov_18/raw')
    from content.data_pointers.lavi_november_18.jackal import jackal_18 as ugv_pointers
    for bag_name, bag_descriptor in ugv_pointers.items():
        experiment = Gmapping(name='gmapping_on_nov_%s' % bag_name, data_sources=bag_descriptor.path, working_dir=execution_dir)
        experiment.run(repetitions=1, periodic_map_saving=True)

    config.base_raw_data_path = os.path.join(config.root_dir_path, 'resources/lavi_apr_18/raw')
    from content.data_pointers.lavi_april_18.jackal import jackal_18 as ugv_pointers
    for bag_name, bag_descriptor in ugv_pointers.items():
        experiment = Gmapping(name='gmapping_on_apr_%s' % bag_name, data_sources=bag_descriptor.path, working_dir=execution_dir)
        experiment.run(repetitions=1, periodic_map_saving=True)

    config.base_raw_data_path = os.path.join(config.root_dir_path, 'resources/lab_maneuvers/raw')
    from content.data_pointers.lab.jackal import jackal_18 as ugv_pointers
    for bag_name, bag_descriptor in ugv_pointers.items():
        experiment = Gmapping(name='gmapping_on_lab_%s' % bag_name, data_sources=bag_descriptor.path, working_dir=execution_dir)
        experiment.run(repetitions=1, periodic_map_saving=True)