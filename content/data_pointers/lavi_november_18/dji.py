import os
from framework import config
from content.data_pointers import lavi_november_18
from framework.data_descriptor import DataDescriptor

base_path = lavi_november_18.base_raw_data_path
markers_locations_path = lavi_november_18.markers_locations_path
base_resources_path = lavi_november_18.base_resources_path

plot1_snapshots_60_meters = {
                        '10-08-5': DataDescriptor(os.path.join(base_path, 'dji', 'DJI_0229.JPG'), '4 landmarks, 60 meters'),
                        '10-08-6': DataDescriptor(os.path.join(base_path, 'dji', 'DJI_0230.JPG'), '4 landmarks, 60 meters'),
                        '10-08-7': DataDescriptor(os.path.join(base_path, 'dji', 'DJI_0231.JPG'), '4 landmarks, 60 meters'),
                        '10-08-8': DataDescriptor(os.path.join(base_path, 'dji', 'DJI_0232.JPG'), '4 landmarks, 60 meters'),
                        '10-08-9': DataDescriptor(os.path.join(base_path, 'dji', 'DJI_0233.JPG'), '4 landmarks, 60 meters'),
                        '10-08-10': DataDescriptor(os.path.join(base_path, 'dji', 'DJI_0234.JPG'), '4 landmarks, 60 meters'),
                    }
plot1_snapshots_60_meters_markers_locations_json_path = os.path.join(markers_locations_path, 'plot1_snapshots_60_meters.json')

plot1_snapshots_80_meters = {
                        '10-07-1': DataDescriptor(os.path.join(base_path, 'dji', 'DJI_0216.JPG'), '4 landmarks, 80 meters'),
                        '10-07-2': DataDescriptor(os.path.join(base_path, 'dji', 'DJI_0217.JPG'), '4 landmarks, 80 meters'),
                        '10-07-3': DataDescriptor(os.path.join(base_path, 'dji', 'DJI_0218.JPG'), '4 landmarks, 80 meters'),
                        '10-07-4': DataDescriptor(os.path.join(base_path, 'dji', 'DJI_0219.JPG'), '4 landmarks, 80 meters'),
                        '10-07-5': DataDescriptor(os.path.join(base_path, 'dji', 'DJI_0220.JPG'), '4 landmarks, 80 meters'),
                        '10-07-6': DataDescriptor(os.path.join(base_path, 'dji', 'DJI_0221.JPG'), '4 landmarks, 80 meters'),
                        '10-07-7': DataDescriptor(os.path.join(base_path, 'dji', 'DJI_0222.JPG'), '4 landmarks, 80 meters'),
                        '10-07-8': DataDescriptor(os.path.join(base_path, 'dji', 'DJI_0223.JPG'), '4 landmarks, 80 meters'),
                        '10-07-9': DataDescriptor(os.path.join(base_path, 'dji', 'DJI_0224.JPG'), '4 landmarks, 80 meters'),
                        '10-07-10': DataDescriptor(os.path.join(base_path, 'dji', 'DJI_0225.JPG'), '4 landmarks, 80 meters'),
                        '10-07-11': DataDescriptor(os.path.join(base_path, 'dji', 'DJI_0226.JPG'), '4 landmarks, 80 meters'),
                        '10-08-1': DataDescriptor(os.path.join(base_path, 'dji', 'DJI_0227.JPG'), '4 landmarks, 80 meters'),
                        '10-08-2': DataDescriptor(os.path.join(base_path, 'dji', 'DJI_0228.JPG'), '4 landmarks, 80 meters'),

                        '10-09-1': DataDescriptor(os.path.join(base_path, 'dji', 'DJI_0235.JPG'), '4 landmarks, 80 meters'),
                        '10-09-2': DataDescriptor(os.path.join(base_path, 'dji', 'DJI_0236.JPG'), '4 landmarks, 80 meters'),
                        '10-09-3': DataDescriptor(os.path.join(base_path, 'dji', 'DJI_0237.JPG'), '4 landmarks, 80 meters'),
                        '10-09-4': DataDescriptor(os.path.join(base_path, 'dji', 'DJI_0238.JPG'), '4 landmarks, 80 meters'),
                        '10-09-5': DataDescriptor(os.path.join(base_path, 'dji', 'DJI_0239.JPG'), '4 landmarks, 80 meters'),
                        '10-09-6': DataDescriptor(os.path.join(base_path, 'dji', 'DJI_0240.JPG'), '4 landmarks, 80 meters'),
                        '10-09-7': DataDescriptor(os.path.join(base_path, 'dji', 'DJI_0241.JPG'), '4 landmarks, 80 meters'),
                        '10-09-8': DataDescriptor(os.path.join(base_path, 'dji', 'DJI_0242.JPG'), '4 landmarks, 80 meters'),
                        '10-09-9': DataDescriptor(os.path.join(base_path, 'dji', 'DJI_0243.JPG'), '4 landmarks, 80 meters'),
                        '10-09-10': DataDescriptor(os.path.join(base_path, 'dji', 'DJI_0244.JPG'), '4 landmarks, 80 meters'),
                        '10-09-11': DataDescriptor(os.path.join(base_path, 'dji', 'DJI_0245.JPG'), '4 landmarks, 80 meters'),
                        '10-09-12': DataDescriptor(os.path.join(base_path, 'dji', 'DJI_0246.JPG'), '4 landmarks, 80 meters'),
                        '10-09-13': DataDescriptor(os.path.join(base_path, 'dji', 'DJI_0247.JPG'), '4 landmarks, 80 meters'),
                        '10-10-1': DataDescriptor(os.path.join(base_path, 'dji', 'DJI_0248.JPG'), '4 landmarks, 80 meters'),
                        '10-10-2': DataDescriptor(os.path.join(base_path, 'dji', 'DJI_0249.JPG'), '4 landmarks, 80 meters'),
                        '10-10-3': DataDescriptor(os.path.join(base_path, 'dji', 'DJI_0250.JPG'), '4 landmarks, 80 meters'),
                        '10-10-4': DataDescriptor(os.path.join(base_path, 'dji', 'DJI_0251.JPG'), '4 landmarks, 80 meters'),
                        '10-10-5': DataDescriptor(os.path.join(base_path, 'dji', 'DJI_0252.JPG'), '4 landmarks, 80 meters'),
                        '10-10-6': DataDescriptor(os.path.join(base_path, 'dji', 'DJI_0253.JPG'), '4 landmarks, 80 meters'),
                        '10-10-7': DataDescriptor(os.path.join(base_path, 'dji', 'DJI_0254.JPG'), '4 landmarks, 80 meters'),
                        '10-10-8': DataDescriptor(os.path.join(base_path, 'dji', 'DJI_0255.JPG'), '4 landmarks, 80 meters'),
                        '10-10-9': DataDescriptor(os.path.join(base_path, 'dji', 'DJI_0256.JPG'), '4 landmarks, 80 meters'),
                        '10-10-10': DataDescriptor(os.path.join(base_path, 'dji', 'DJI_0257.JPG'), '4 landmarks, 80 meters'),
                        '10-10-11': DataDescriptor(os.path.join(base_path, 'dji', 'DJI_0258.JPG'), '4 landmarks, 80 meters'),
                        '10-10-12': DataDescriptor(os.path.join(base_path, 'dji', 'DJI_0259.JPG'), '4 landmarks, 80 meters'),
                        '10-10-13': DataDescriptor(os.path.join(base_path, 'dji', 'DJI_0260.JPG'), '4 landmarks, 80 meters'),
                        '10-10-14': DataDescriptor(os.path.join(base_path, 'dji', 'DJI_0261.JPG'), '4 landmarks, 80 meters'),
                        '10-11-1': DataDescriptor(os.path.join(base_path, 'dji', 'DJI_0262.JPG'), '4 landmarks, 80 meters'),
                        '10-11-2': DataDescriptor(os.path.join(base_path, 'dji', 'DJI_0263.JPG'), '4 landmarks, 80 meters'),
                        '10-11-3': DataDescriptor(os.path.join(base_path, 'dji', 'DJI_0264.JPG'), '4 landmarks, 80 meters'),
                        '10-12-1': DataDescriptor(os.path.join(base_path, 'dji', 'DJI_0265.JPG'), '4 landmarks, 80 meters'),
                        '10-12-2': DataDescriptor(os.path.join(base_path, 'dji', 'DJI_0266.JPG'), '4 landmarks, 80 meters'),
                        '10-13-1': DataDescriptor(os.path.join(base_path, 'dji', 'DJI_0267.JPG'), '4 landmarks, 80 meters'),
                        '10-13-2': DataDescriptor(os.path.join(base_path, 'dji', 'DJI_0268.JPG'), '4 landmarks, 80 meters'),
                        '10-14-1': DataDescriptor(os.path.join(base_path, 'dji', 'DJI_0269.JPG'), '4 landmarks, 80 meters'),
                        '10-14-2': DataDescriptor(os.path.join(base_path, 'dji', 'DJI_0270.JPG'), '4 landmarks, 80 meters'),
                        '10-15-1': DataDescriptor(os.path.join(base_path, 'dji', 'DJI_0271.JPG'), '4 landmarks, 80 meters'),
                        '10-15-2': DataDescriptor(os.path.join(base_path, 'dji', 'DJI_0272.JPG'), '4 landmarks, 80 meters'),
                        '10-15-3': DataDescriptor(os.path.join(base_path, 'dji', 'DJI_0273.JPG'), '4 landmarks, 80 meters'),
                        '10-16-1': DataDescriptor(os.path.join(base_path, 'dji', 'DJI_0274.JPG'), '4 landmarks, 80 meters'),
                        '10-16-2': DataDescriptor(os.path.join(base_path, 'dji', 'DJI_0275.JPG'), '4 landmarks, 80 meters'),
                        '10-16-3': DataDescriptor(os.path.join(base_path, 'dji', 'DJI_0276.JPG'), '4 landmarks, 80 meters'),
                        '10-17-1': DataDescriptor(os.path.join(base_path, 'dji', 'DJI_0277.JPG'), '4 landmarks, 80 meters'),

                        '10-26-1': DataDescriptor(os.path.join(base_path, 'dji', 'DJI_0282.JPG'), '4 landmarks, 80 meters, with vehicle'),
                        '10-26-2': DataDescriptor(os.path.join(base_path, 'dji', 'DJI_0283.JPG'), '4 landmarks, 80 meters, with vehicle'),
                        '10-26-3': DataDescriptor(os.path.join(base_path, 'dji', 'DJI_0284.JPG'), '4 landmarks, 80 meters, with vehicle'),
                        '10-27-1': DataDescriptor(os.path.join(base_path, 'dji', 'DJI_0285.JPG'), '4 landmarks, 80 meters, with vehicle'),
                        '10-27-2': DataDescriptor(os.path.join(base_path, 'dji', 'DJI_0286.JPG'), '4 landmarks, 80 meters, with vehicle'),
                        '10-28-1': DataDescriptor(os.path.join(base_path, 'dji', 'DJI_0287.JPG'), '4 landmarks, 80 meters, with vehicle'),
                        '10-29-1': DataDescriptor(os.path.join(base_path, 'dji', 'DJI_0288.JPG'), '4 landmarks, 80 meters, with vehicle'),
                        '10-29-2': DataDescriptor(os.path.join(base_path, 'dji', 'DJI_0289.JPG'), '4 landmarks, 80 meters, with vehicle'),
                        '10-30-1': DataDescriptor(os.path.join(base_path, 'dji', 'DJI_0290.JPG'), '4 landmarks, 80 meters, with vehicle'),
                        '10-30-2': DataDescriptor(os.path.join(base_path, 'dji', 'DJI_0291.JPG'), '4 landmarks, 80 meters, with vehicle'),
                        '10-31-1': DataDescriptor(os.path.join(base_path, 'dji', 'DJI_0292.JPG'), '4 landmarks, 80 meters, with vehicle'),
                        '10-32-1': DataDescriptor(os.path.join(base_path, 'dji', 'DJI_0293.JPG'), '4 landmarks, 80 meters, with vehicle'),
                    }
plot1_snapshots_80_meters_markers_locations_json_path = os.path.join(markers_locations_path, 'plot1_snapshots_80_meters.json')
plot1_snapshots_80_meters_ugv_poses_path = os.path.join(base_resources_path, 'ugv_poses', 'plot1_snapshots_80_meters_ugv_poses.json')

plot2_snapshots_80_meters = {
                        '11-06-1': DataDescriptor(os.path.join(base_path, 'dji', 'DJI_0295.JPG'), '4 landmarks, 80 meters'),
                        '11-07-1': DataDescriptor(os.path.join(base_path, 'dji', 'DJI_0296.JPG'), '4 landmarks, 80 meters'),
                        '11-07-2': DataDescriptor(os.path.join(base_path, 'dji', 'DJI_0297.JPG'), '4 landmarks, 80 meters'),
                        '11-07-3': DataDescriptor(os.path.join(base_path, 'dji', 'DJI_0298.JPG'), '4 landmarks, 80 meters'),
                        '11-07-4': DataDescriptor(os.path.join(base_path, 'dji', 'DJI_0299.JPG'), '4 landmarks, 80 meters'),
                        '11-07-5': DataDescriptor(os.path.join(base_path, 'dji', 'DJI_0300.JPG'), '4 landmarks, 80 meters'),
                        '11-07-6': DataDescriptor(os.path.join(base_path, 'dji', 'DJI_0301.JPG'), '4 landmarks, 80 meters'),
                        '11-07-7': DataDescriptor(os.path.join(base_path, 'dji', 'DJI_0302.JPG'), '4 landmarks, 80 meters'),
                        '11-07-8': DataDescriptor(os.path.join(base_path, 'dji', 'DJI_0303.JPG'), '4 landmarks, 80 meters'),
                        '11-07-9': DataDescriptor(os.path.join(base_path, 'dji', 'DJI_0304.JPG'), '4 landmarks, 80 meters'),
                        '11-07-11': DataDescriptor(os.path.join(base_path, 'dji', 'DJI_0309.JPG'), '4 landmarks, 80 meters'),
                        '11-07-12': DataDescriptor(os.path.join(base_path, 'dji', 'DJI_0310.JPG'), '4 landmarks, 80 meters'),
                        '11-07-13': DataDescriptor(os.path.join(base_path, 'dji', 'DJI_0311.JPG'), '4 landmarks, 80 meters'),
                        '11-07-14': DataDescriptor(os.path.join(base_path, 'dji', 'DJI_0312.JPG'), '4 landmarks, 80 meters'),
                        '11-07-15': DataDescriptor(os.path.join(base_path, 'dji', 'DJI_0313.JPG'), '4 landmarks, 80 meters'),
                        '11-07-16': DataDescriptor(os.path.join(base_path, 'dji', 'DJI_0314.JPG'), '4 landmarks, 80 meters'),
                        '11-07-17': DataDescriptor(os.path.join(base_path, 'dji', 'DJI_0315.JPG'), '4 landmarks, 80 meters'),
                        '11-07-18': DataDescriptor(os.path.join(base_path, 'dji', 'DJI_0316.JPG'), '4 landmarks, 80 meters'),
                        '11-07-19': DataDescriptor(os.path.join(base_path, 'dji', 'DJI_0317.JPG'), '4 landmarks, 80 meters'),
                        '11-07-20': DataDescriptor(os.path.join(base_path, 'dji', 'DJI_0318.JPG'), '4 landmarks, 80 meters'),
                        '11-07-21': DataDescriptor(os.path.join(base_path, 'dji', 'DJI_0319.JPG'), '4 landmarks, 80 meters'),
                        '11-07-22': DataDescriptor(os.path.join(base_path, 'dji', 'DJI_0320.JPG'), '4 landmarks, 80 meters'),
                    }
plot2_snapshots_80_meters_markers_locations_json_path = os.path.join(markers_locations_path, 'plot2_snapshots_80_meters.json')

plot3_snapshots_60_meters = {
                        '11-30-4': DataDescriptor(os.path.join(base_path, 'dji', 'DJI_0345.JPG'), '4 landmarks, 80 meters'),
                        '11-30-5': DataDescriptor(os.path.join(base_path, 'dji', 'DJI_0346.JPG'), '4 landmarks, 80 meters'),
                        '11-31-1': DataDescriptor(os.path.join(base_path, 'dji', 'DJI_0347.JPG'), '4 landmarks, 80 meters'),
                        '11-31-2': DataDescriptor(os.path.join(base_path, 'dji', 'DJI_0348.JPG'), '4 landmarks, 80 meters'),
                        '11-31-3': DataDescriptor(os.path.join(base_path, 'dji', 'DJI_0349.JPG'), '4 landmarks, 80 meters'),
                        '11-31-4': DataDescriptor(os.path.join(base_path, 'dji', 'DJI_0350.JPG'), '4 landmarks, 80 meters'),
                        '11-31-5': DataDescriptor(os.path.join(base_path, 'dji', 'DJI_0352.JPG'), '4 landmarks, 80 meters'),
                        '11-31-6': DataDescriptor(os.path.join(base_path, 'dji', 'DJI_0353.JPG'), '4 landmarks, 80 meters'),
                        '11-31-7': DataDescriptor(os.path.join(base_path, 'dji', 'DJI_0354.JPG'), '4 landmarks, 80 meters'),
                    }
plot3_snapshots_60_meters_markers_locations_json_path = os.path.join(markers_locations_path, 'plot3_snapshots_60_meters.json')


plot3_snapshots_80_meters = {
                        '11-28-1': DataDescriptor(os.path.join(base_path, 'dji', 'DJI_0323.JPG'), '4 landmarks, 80 meters'),
                        '11-28-2': DataDescriptor(os.path.join(base_path, 'dji', 'DJI_0324.JPG'), '4 landmarks, 80 meters'),
                        '11-28-3': DataDescriptor(os.path.join(base_path, 'dji', 'DJI_0325.JPG'), '4 landmarks, 80 meters'),
                        '11-28-4': DataDescriptor(os.path.join(base_path, 'dji', 'DJI_0326.JPG'), '4 landmarks, 80 meters'),
                        '11-28-5': DataDescriptor(os.path.join(base_path, 'dji', 'DJI_0327.JPG'), '4 landmarks, 80 meters'),
                        '11-28-6': DataDescriptor(os.path.join(base_path, 'dji', 'DJI_0328.JPG'), '4 landmarks, 80 meters'),
                        '11-29-1': DataDescriptor(os.path.join(base_path, 'dji', 'DJI_0328.JPG'), '4 landmarks, 80 meters'),
                        '11-29-2': DataDescriptor(os.path.join(base_path, 'dji', 'DJI_0329.JPG'), '4 landmarks, 80 meters'),
                        '11-29-3': DataDescriptor(os.path.join(base_path, 'dji', 'DJI_0330.JPG'), '4 landmarks, 80 meters'),
                        '11-29-4': DataDescriptor(os.path.join(base_path, 'dji', 'DJI_0331.JPG'), '4 landmarks, 80 meters'),
                        '11-29-5': DataDescriptor(os.path.join(base_path, 'dji', 'DJI_0332.JPG'), '4 landmarks, 80 meters'),
                        '11-29-6': DataDescriptor(os.path.join(base_path, 'dji', 'DJI_0333.JPG'), '4 landmarks, 80 meters'),
                        '11-29-7': DataDescriptor(os.path.join(base_path, 'dji', 'DJI_0334.JPG'), '4 landmarks, 80 meters'),
                        '11-29-8': DataDescriptor(os.path.join(base_path, 'dji', 'DJI_0335.JPG'), '4 landmarks, 80 meters'),
                        '11-29-9': DataDescriptor(os.path.join(base_path, 'dji', 'DJI_0336.JPG'), '4 landmarks, 80 meters'),
                        '11-29-10': DataDescriptor(os.path.join(base_path, 'dji', 'DJI_0337.JPG'), '4 landmarks, 80 meters'),
                        '11-29-11': DataDescriptor(os.path.join(base_path, 'dji', 'DJI_0338.JPG'), '4 landmarks, 80 meters'),
                        '11-29-12': DataDescriptor(os.path.join(base_path, 'dji', 'DJI_0339.JPG'), '4 landmarks, 80 meters'),
                        '11-29-13': DataDescriptor(os.path.join(base_path, 'dji', 'DJI_0340.JPG'), '4 landmarks, 80 meters'),
                        '11-29-14': DataDescriptor(os.path.join(base_path, 'dji', 'DJI_0341.JPG'), '4 landmarks, 80 meters'),
                        '11-30-1': DataDescriptor(os.path.join(base_path, 'dji', 'DJI_0342.JPG'), '4 landmarks, 80 meters'),
                        '11-30-2': DataDescriptor(os.path.join(base_path, 'dji', 'DJI_0343.JPG'), '4 landmarks, 80 meters'),
                        '11-30-3': DataDescriptor(os.path.join(base_path, 'dji', 'DJI_0344.JPG'), '4 landmarks, 80 meters'),
                    }
plot3_snapshots_80_meters_markers_locations_json_path = os.path.join(markers_locations_path, 'plot3_snapshots_80_meters.json')


plot4_snapshots_80_meters = {
                        '11-42-1': DataDescriptor(os.path.join(base_path, 'dji', 'DJI_0357.JPG'), '4 landmarks, 80 meters'),
                        '11-42-2': DataDescriptor(os.path.join(base_path, 'dji', 'DJI_0358.JPG'), '4 landmarks, 80 meters'),
                        '11-42-3': DataDescriptor(os.path.join(base_path, 'dji', 'DJI_0359.JPG'), '4 landmarks, 80 meters'),
                        '11-42-4': DataDescriptor(os.path.join(base_path, 'dji', 'DJI_0360.JPG'), '4 landmarks, 80 meters'),
                        '11-42-5': DataDescriptor(os.path.join(base_path, 'dji', 'DJI_0361.JPG'), '4 landmarks, 80 meters'),
                        '11-42-6': DataDescriptor(os.path.join(base_path, 'dji', 'DJI_0362.JPG'), '4 landmarks, 80 meters'),
                        '11-42-7': DataDescriptor(os.path.join(base_path, 'dji', 'DJI_0363.JPG'), '4 landmarks, 80 meters'),
                        '11-42-8': DataDescriptor(os.path.join(base_path, 'dji', 'DJI_0364.JPG'), '4 landmarks, 80 meters'),
                        '11-43-1': DataDescriptor(os.path.join(base_path, 'dji', 'DJI_0365.JPG'), '4 landmarks, 80 meters'),
                        '11-43-2': DataDescriptor(os.path.join(base_path, 'dji', 'DJI_0366.JPG'), '4 landmarks, 80 meters'),
                        '11-43-3': DataDescriptor(os.path.join(base_path, 'dji', 'DJI_0367.JPG'), '4 landmarks, 80 meters'),
                        '11-43-4': DataDescriptor(os.path.join(base_path, 'dji', 'DJI_0368.JPG'), '4 landmarks, 80 meters'),
                        '11-43-5': DataDescriptor(os.path.join(base_path, 'dji', 'DJI_0369.JPG'), '4 landmarks, 80 meters'),
                        '11-43-6': DataDescriptor(os.path.join(base_path, 'dji', 'DJI_0370.JPG'), '4 landmarks, 80 meters'),
                        '11-43-7': DataDescriptor(os.path.join(base_path, 'dji', 'DJI_0371.JPG'), '4 landmarks, 80 meters'),
                        '11-43-8': DataDescriptor(os.path.join(base_path, 'dji', 'DJI_0372.JPG'), '4 landmarks, 80 meters'),
                        '11-43-9': DataDescriptor(os.path.join(base_path, 'dji', 'DJI_0373.JPG'), '4 landmarks, 80 meters'),
                        '11-43-10': DataDescriptor(os.path.join(base_path, 'dji', 'DJI_0374.JPG'), '4 landmarks, 80 meters'),
                        '11-43-11': DataDescriptor(os.path.join(base_path, 'dji', 'DJI_0375.JPG'), '4 landmarks, 80 meters'),
                        '11-43-12': DataDescriptor(os.path.join(base_path, 'dji', 'DJI_0376.JPG'), '4 landmarks, 80 meters'),
                        '11-43-13': DataDescriptor(os.path.join(base_path, 'dji', 'DJI_0377.JPG'), '4 landmarks, 80 meters'),
                        '11-43-14': DataDescriptor(os.path.join(base_path, 'dji', 'DJI_0378.JPG'), '4 landmarks, 80 meters'),
                        '11-43-15': DataDescriptor(os.path.join(base_path, 'dji', 'DJI_0379.JPG'), '4 landmarks, 80 meters'),
                        '11-43-16': DataDescriptor(os.path.join(base_path, 'dji', 'DJI_0380.JPG'), '4 landmarks, 80 meters'),
                        '11-43-17': DataDescriptor(os.path.join(base_path, 'dji', 'DJI_0381.JPG'), '4 landmarks, 80 meters'),
                        '11-43-18': DataDescriptor(os.path.join(base_path, 'dji', 'DJI_0382.JPG'), '4 landmarks, 80 meters'),
                        '11-44-1': DataDescriptor(os.path.join(base_path, 'dji', 'DJI_0383.JPG'), '4 landmarks, 80 meters'),
                    }
plot4_snapshots_80_meters_markers_locations_json_path = os.path.join(markers_locations_path, 'plot4_snapshots_80_meters.json')

trunks_detection_results_dir = os.path.join(config.base_results_path, 'trunks_detection')
plot1_selected_trunks_detection_experiments = ['trunks_detection_on_nov1_10-09-9', 'trunks_detection_on_nov1_10-10-9',
                                               'trunks_detection_on_nov1_10-12-1']
plot2_selected_trunks_detection_experiments = ['trunks_detection_on_nov2_11-07-5', 'trunks_detection_on_nov2_11-07-19',
                                               'trunks_detection_on_nov2_11-07-22']
