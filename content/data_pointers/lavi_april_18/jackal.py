import os
import experiments_framework.framework.config as config
from experiments_framework.framework.data_descriptor import DataDescriptor

base_path = config.base_data_path

jackal_18 = {'15-09-35': DataDescriptor(os.path.join(base_path, 'jackal_18', '2018-04-24-15-09-35_0.bag'), 'short warm up'),
             '15-11-01': DataDescriptor(os.path.join(base_path, 'jackal_18', '2018-04-24-15-11-01_0.bag'), 'fork in rows 3-4, 4-5 and 5-6 ###############'),
             '17-09-14': DataDescriptor(os.path.join(base_path, 'jackal_18', '2018-04-24-17-09-14_0.bag'), 'fork in rows 3-4, 4-5 and 5-6 (experiment 5.4) ###############'),
             '17-18-53': DataDescriptor(os.path.join(base_path, 'jackal_18', '2018-04-24-17-18-53_0.bag'), 'fork in rows 3-4, 4-5 and 5-6 (experiment 5.4) ###############'),
             '17-45-36': DataDescriptor(os.path.join(base_path, 'jackal_18', '2018-04-24-17-45-36_0.bag'), 'fork in rows 3-4, 4-5 and 5-6 (experiment 5.4) ###############'),
             '17-58-58': DataDescriptor(os.path.join(base_path, 'jackal_18', '2018-04-24-17-58-58_0.bag'), 'random trajectory ###############'),
             '18-24-26': DataDescriptor((os.path.join(base_path, 'jackal_18', '2018-04-24-18-24-26_0.bag'),
                                         os.path.join(base_path, 'jackal_18', '2018-04-24-18-28-23_1.bag'),
                                         os.path.join(base_path, 'jackal_18', '2018-04-24-18-30-40_2.bag')), 'mapping ###############')}

jackal_19 = [
            ]

forks = [(name, descriptor) for (name, descriptor) in jackal_18.iteritems() if name in ['15-11-01', '17-09-14', '17-18-53', '17-45-36']]