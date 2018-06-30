import os
import experiments_framework.framework.config as config
from experiments_framework.framework.data_descriptor import DataDescriptor

base_path = config.panorama_path

full_orchard = {
                'dji_afternoon': DataDescriptor(os.path.join(base_path, 'DJI_0178_afternoon_good_1380x640_stitch_full_movie.jpg'), 'XXXXXX'),
              }