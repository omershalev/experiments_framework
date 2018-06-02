import numpy as np
import pickle
import cv2

import experiments_framework.framework.cv_utils as cv_utils

if __name__ == '__main__':

    import experiments_framework.content.data_pointers.lavi_april_18.dji as dji_data

    samples = []
    try:
        for snapshot_name, snapshot_descriptor in dji_data.snapshots_60_meters.items() + dji_data.snapshots_80_meters.items():
            img = cv2.imread(snapshot_descriptor.path)

            samples += cv_utils.sample_hsv_color(img)
    finally:
        with open(r'/home/omer/Downloads/hsv.pkl', 'wb') as p:
            pickle.dump(samples, p)
    h_mean = np.mean([sample[0] for sample in samples])
    h_median = np.median([sample[0] for sample in samples])
    h_std = np.std([sample[0] for sample in samples])
    print ('H: (mean, median, std) = %s' % str((h_mean, h_median, h_std)))
    print ('H-colorizer: (mean, median, std) = %s' % str((2*h_mean, 2*h_median, 2*h_std)))
    s_mean = np.mean([sample[1] for sample in samples])
    s_median = np.median([sample[1] for sample in samples])
    s_std = np.std([sample[1] for sample in samples])
    print ('S: (mean, median, std) = %s' % str((s_mean, s_median, s_std)))
    print ('S-colorizer: (mean, median, std) = %s' % str((100*s_mean/256.0, 100*s_median/256.0, 100*s_std/256.0)))
    v_mean = np.mean([sample[2] for sample in samples])
    v_median = np.median([sample[2] for sample in samples])
    v_std = np.std([sample[2] for sample in samples])
    print ('V: (mean, median, std) = %s' % str((v_mean, v_median, v_std)))
    print ('V: (mean, median, std) = %s' % str((100*v_mean/256.0, 100*v_median/256.0, 100*v_std/256.0)))
