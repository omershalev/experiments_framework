from framework import utils
from content.experiments.jackal_gmapping import JackalGmapping

#################################################################################################
#                                             CONFIG                                            #
#################################################################################################
ugv_bag_name = '18-24-26'
periodic_map_saving = False
setup = 'apr' # apr / nov / lab
#################################################################################################

if setup == 'apr':
    from content.data_pointers.lavi_april_18.jackal import jackal_18 as ugv_pointers
elif setup == 'nov':
    raise NotImplementedError
elif setup == 'lab':
    raise NotImplementedError

if __name__ == '__main__':

    execution_dir = utils.create_new_execution_folder('jackal_gmapping')
    bag_descriptor = ugv_pointers[ugv_bag_name]
    experiment = JackalGmapping(name='gmapping_on_apr_%s' % ugv_bag_name, data_sources=bag_descriptor.path, working_dir=execution_dir)
    experiment.run(repetitions=1, periodic_map_saving=periodic_map_saving)
