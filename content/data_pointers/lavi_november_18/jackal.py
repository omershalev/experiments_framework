import os
from content.data_pointers import lavi_november_18
from framework.data_descriptor import DataDescriptor

base_path = lavi_november_18.base_raw_data_path

jackal_18 = {'10-09': DataDescriptor(os.path.join(base_path, 'jackal', '2018-11-01-10-09-28_0.bag'), 'random'),
             '10-25': DataDescriptor(os.path.join(base_path, 'jackal', '2018-11-01-10-25-21_0.bag'), 's-shape, coupled with aerial observations')}