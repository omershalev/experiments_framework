import numpy as np
import cv2
import json

from experiments_framework.content.data_pointers.lavi_april_18 import dji
from experiments_framework.framework import viz_utils

if __name__ == '__main__':
    key1, data_descriptor1 = dji.snapshots_60_meters.items()[0]
    key2, data_descriptor2 = dji.snapshots_60_meters.items()[1]
    image1 = cv2.imread(data_descriptor1.path)
    image2 = cv2.imread(data_descriptor2.path)
    with open('/home/omer/orchards_ws/data/lavi_apr_18/markers_locations/snapshots_60_meters.json') as f:
        markers_locations = json.load(f)
    locations1 = markers_locations[key1]
    locations2 = markers_locations[key2]

    # viz_utils.show_image('image1', image1)
    # viz_utils.show_image('image2', image2)

    r1 = cv2.estimateRigidTransform(np.array(locations1), np.array(locations2), fullAffine=True)
    r2 = cv2.estimateRigidTransform(np.array(locations1), np.array(locations2), fullAffine=False)
    r3 = cv2.estimateAffine2D(np.array(locations1), np.array(locations2))
    r4 = cv2.estimateAffinePartial2D(np.array(locations1), np.array(locations2))

    print (r1, r2, r3, r4)
    print ('hi')
