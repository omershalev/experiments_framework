import time
import datetime

from experiments_framework.framework.experiment import Experiment
import experiments_framework.framework.utils as utils
import experiments_framework.content.jackal_data as jackal_data

class GmappingExperiment(Experiment):
    def task(self):

        utils.ros_start(use_sim_time=True)
        utils.ros_launch('jackal_navigation', 'gmapping.launch')
        # _, bag_duration = utils.ros_bag_play(r'/home/omer/orchards_ws/data/jackal_18/2018-04-24-18-30-40_2.bag', use_clock=True)
        _, bag_duration = utils.ros_bag_play(self.data_sources, use_clock=True)
        start_time = datetime.datetime.now()
        while True:
            if (datetime.datetime.now() - start_time).seconds > (bag_duration + 1):
                break
            utils.ros_save_map(map_name=datetime.datetime.now().strftime('map_%H_%M_%S'),
                               dir_name=self.repetition_dir)
            time.sleep(3)

        # bag_proc.kill()
        # gmapping_proc.kill()
        # ros_proc.kill()


        # p2 = subprocess.Popen(['rosbag', 'play', '/home/omer/orchards_ws/data/jackal_18/2018-04-04-17-26-09_0.bag'])

if __name__ == '__main__':
    # utils.ros_kill_all()
    #
    # ros_proc = utils.ros_start(use_sim_time=True)
    # gmapping_proc = utils.ros_launch('jackal_navigation', 'gmapping.launch')
    # bag_proc, bag_duration = utils.ros_bag_play(r'/home/omer/orchards_ws/data/jackal_18/2018-04-24-18-30-40_2.bag', use_clock=True)
    # start_time = datetime.datetime.now()
    # while True:
    #     print datetime.datetime.now()
    #     if (datetime.datetime.now() - start_time).seconds > (bag_duration + 1):
    #         break
    #     utils.ros_save_map(map_name=datetime.datetime.now().strftime('map_%H_%M_%S'), dir_name=config.base_output_path)
    #     time.sleep(4)
    #
    # bag_proc.kill()
    # gmapping_proc.kill()
    # ros_proc.kill()
    #
    # utils.ros_kill_all()

    import os
    import experiments_framework.framework.config as config
    execution_name = 'gmapping_jackal_18'
    execution_dir_name = '%s_%s' % (datetime.datetime.now().strftime('%Y%m%d-%H%M%S'), execution_name)
    execution_dir = os.path.join(config.base_output_path, execution_dir_name)
    os.mkdir(execution_dir)

    for bag_file in jackal_data.jackal_18:
        experiment = GmappingExperiment(name='gmapping', data_sources=bag_file, working_dir=execution_dir)
        experiment.run(2)