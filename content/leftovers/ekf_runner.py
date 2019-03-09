from framework import utils
from content.leftovers.ekf import Ekf

if __name__ == '__main__':

    execution_dir = utils.create_new_execution_folder('jackal_ekf')

    from content.data_pointers.lavi_april_18.jackal import jackal_18 as ugv_pointers
    bag_name = '18-24-26'
    bag_descriptor = ugv_pointers[bag_name]
    experiment = Ekf(name='ekf_on_apr_%s' % bag_name, data_sources=bag_descriptor.path, working_dir=execution_dir)
    experiment.run(repetitions=1, odom=False, ekf=True, gps=False)
    experiment = Ekf(name='ekf_gps_on_apr_%s' % bag_name, data_sources=bag_descriptor.path, working_dir=execution_dir)
    experiment.run(repetitions=1, odom=False, ekf=True, gps=True)


    # config.base_raw_data_path = os.path.join(config.root_dir_path, 'resources/lab_maneuvers/raw')
    # from content.data_pointers.lab.jackal import jackal_18 as ugv_pointers
    # for bag_name, bag_descriptor in ugv_pointers.items():
    #     experiment = Ekf(name='ekf_odom_on_lab_%s' % bag_name, data_sources=bag_descriptor.path, working_dir=execution_dir)
    #     experiment.run(repetitions=1, odom=True, ekf=True, gps=False)
    #     experiment = Ekf(name='ekf_on_lab_%s' % bag_name, data_sources=bag_descriptor.path, working_dir=execution_dir)
    #     experiment.run(repetitions=1, odom=True, ekf=True, gps=False)
