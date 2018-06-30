import os
import experiments_framework.framework.config as config
from experiments_framework.framework.data_descriptor import DataDescriptor

base_path = config.base_raw_data_path

snapshots_60_meters = {
                        '15-20-1': DataDescriptor(os.path.join(base_path, 'dji', 'DJI_0137.JPG'), 'table in 5-6, 4 landmarks, 60? meters'),
                        '15-20-2': DataDescriptor(os.path.join(base_path, 'dji', 'DJI_0138.JPG'), 'table in 5-6, 4 landmarks, 60? meters'),
                        '15-20-3': DataDescriptor(os.path.join(base_path, 'dji', 'DJI_0139.JPG'), '4 landmarks, 60? meters'),
                        '15-20-4': DataDescriptor(os.path.join(base_path, 'dji', 'DJI_0140.JPG'), '4 landmarks, 60? meters'),
                        '15-20-5': DataDescriptor(os.path.join(base_path, 'dji', 'DJI_0141.JPG'), '4 landmarks, 60? meters'),

                        '16-54-1': DataDescriptor(os.path.join(base_path, 'dji', 'DJI_0159.JPG'), '4 landmarks, 60 meters'),
                        '16-54-2': DataDescriptor(os.path.join(base_path, 'dji', 'DJI_0160.JPG'), '4 landmarks, 60 meters'),
                        '16-54-3': DataDescriptor(os.path.join(base_path, 'dji', 'DJI_0161.JPG'), '4 landmarks, 60 meters'),

                        '19-03-1': DataDescriptor(os.path.join(base_path, 'dji', 'DJI_0179.JPG'), '4 landmarks, 60? meters'),
                        '19-03-2': DataDescriptor(os.path.join(base_path, 'dji', 'DJI_0180.JPG'), '4 landmarks, 60? meters'),
                        '19-03-3': DataDescriptor(os.path.join(base_path, 'dji', 'DJI_0181.JPG'), '4 landmarks, 60? meters'),
                        '19-03-4': DataDescriptor(os.path.join(base_path, 'dji', 'DJI_0182.JPG'), '4 landmarks, 60? meters'),
                        '19-03-5': DataDescriptor(os.path.join(base_path, 'dji', 'DJI_0183.JPG'), '4 landmarks, 60? meters')
                      }


snapshots_80_meters = {
                         '15-08-1': DataDescriptor(os.path.join(base_path, 'dji', 'DJI_0115.JPG'), '4 landmarks, 80 meters'),
                         '15-08-2': DataDescriptor(os.path.join(base_path, 'dji', 'DJI_0116.JPG'), '4 landmarks, 80 meters'),
                         '15-09-1': DataDescriptor(os.path.join(base_path, 'dji', 'DJI_0117.JPG'), '4 landmarks, 80 meters'),
                         '15-09-2': DataDescriptor(os.path.join(base_path, 'dji', 'DJI_0118.JPG'), '4 landmarks, 80 meters'),
                         '15-09-3': DataDescriptor(os.path.join(base_path, 'dji', 'DJI_0119.JPG'), '4 landmarks, 80 meters'),
                         '15-10-1': DataDescriptor(os.path.join(base_path, 'dji', 'DJI_0120.JPG'), '4 landmarks, 80 meters'),
                         '15-17-1': DataDescriptor(os.path.join(base_path, 'dji', 'DJI_0124.JPG'), 'table in 3-4, 4 landmarks, 80 meters'),
                         '15-18-1': DataDescriptor(os.path.join(base_path, 'dji', 'DJI_0125.JPG'), 'table in 3-4, 4 landmarks, 80 meters'),
                         '15-18-2': DataDescriptor(os.path.join(base_path, 'dji', 'DJI_0126.JPG'), 'table in 3-4, 4 landmarks, 80 meters'),
                         '15-18-3': DataDescriptor(os.path.join(base_path, 'dji', 'DJI_0128.JPG'), 'table in 4-5, 4 landmarks, 80 meters'),
                         '15-18-4': DataDescriptor(os.path.join(base_path, 'dji', 'DJI_0129.JPG'), 'table in 4-5, 4 landmarks, 80 meters'),
                         '15-19-1': DataDescriptor(os.path.join(base_path, 'dji', 'DJI_0132.JPG'), 'table in 5-6, 4 landmarks, 80 meters'),
                         '15-19-2': DataDescriptor(os.path.join(base_path, 'dji', 'DJI_0133.JPG'), 'table in 5-6, 4 landmarks, 80 meters'),
                         '15-19-3': DataDescriptor(os.path.join(base_path, 'dji', 'DJI_0134.JPG'), 'table in 5-6, 4 landmarks, 80 meters'),
                         '15-19-4': DataDescriptor(os.path.join(base_path, 'dji', 'DJI_0135.JPG'), 'table in 5-6, 4 landmarks, 80 meters'),
                         '15-19-5': DataDescriptor(os.path.join(base_path, 'dji', 'DJI_0136.JPG'), 'table in 5-6, 4 landmarks, 80 meters'),

                         '15-53-1': DataDescriptor(os.path.join(base_path, 'dji', 'DJI_0149.JPG'), '4 landmarks, 80? meters'),
                         '15-53-2': DataDescriptor(os.path.join(base_path, 'dji', 'DJI_0150.JPG'), '4 landmarks, 80? meters'),
                         '15-53-3': DataDescriptor(os.path.join(base_path, 'dji', 'DJI_0151.JPG'), '4 landmarks, 80? meters'),
                         '15-53-4': DataDescriptor(os.path.join(base_path, 'dji', 'DJI_0152.JPG'), '4 landmarks, 80? meters'),

                         '16-55-1': DataDescriptor(os.path.join(base_path, 'dji', 'DJI_0162.JPG'), '4 landmarks, 80 meters'),
                         '16-55-2': DataDescriptor(os.path.join(base_path, 'dji', 'DJI_0163.JPG'), '4 landmarks, 80 meters'),
                         '16-55-3': DataDescriptor(os.path.join(base_path, 'dji', 'DJI_0164.JPG'), '4 landmarks, 80 meters'),
                         '16-55-4': DataDescriptor(os.path.join(base_path, 'dji', 'DJI_0165.JPG'), '4 landmarks, 80 meters'),
                         '16-55-5': DataDescriptor(os.path.join(base_path, 'dji', 'DJI_0166.JPG'), '4 landmarks, 80 meters'),

                         '19-04-1': DataDescriptor(os.path.join(base_path, 'dji', 'DJI_0184.JPG'), '4 landmarks, 80? meters'),
                         '19-04-2': DataDescriptor(os.path.join(base_path, 'dji', 'DJI_0185.JPG'), '4 landmarks, 80? meters'),
                         '19-04-3': DataDescriptor(os.path.join(base_path, 'dji', 'DJI_0186.JPG'), '4 landmarks, 80? meters'),
                         '19-04-4': DataDescriptor(os.path.join(base_path, 'dji', 'DJI_0187.JPG'), '4 landmarks, 80? meters')
                      }

