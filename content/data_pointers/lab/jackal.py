import os
import framework.config as config
from framework.data_descriptor import DataDescriptor

base_path = config.base_raw_data_path

jackal_18 = {'9-40': DataDescriptor(os.path.join(base_path, '2018-11-03-09-40-07_0.bag'), 'outdoor'),
             '9-53': DataDescriptor(os.path.join(base_path, '2018-11-03-09-53-36_0.bag'), 'indoor 1'),
             '9-55': DataDescriptor(os.path.join(base_path, '2018-11-03-09-55-03_0.bag'), 'indoor 2'),
             '9-59': DataDescriptor(os.path.join(base_path, '2018-11-03-09-59-58_0.bag'), 'indoor 3'),
             }