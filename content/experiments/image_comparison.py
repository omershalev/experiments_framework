import cv2

import experiments_framework.framework.viz_utils as viz_utils
import air_ground_orchard_navigation.computer_vision.image_alignment as align

if __name__ == '__main__':

    import experiments_framework.content.data_pointers.lavi_april_18.dji as dji_data

    noon = dji_data.snapshots_80_meters['15-08-1']
    # late_noon = dji_data.snapshots_80_meters['15-53-1']
    late_noon = dji_data.snapshots_80_meters['19-04-1']
    # afternoon = dji_data.snapshots_80_meters['16-55-1']
    afternoon = dji_data.snapshots_80_meters['15-09-1']
    # late_afternoon = dji_data.snapshots_80_meters['19-04-1']
    late_afternoon = dji_data.snapshots_80_meters['15-09-2']

    img_noon = cv2.cvtColor(cv2.imread(noon.path), cv2.COLOR_BGR2GRAY)
    img_late_noon = cv2.cvtColor(cv2.imread(late_noon.path), cv2.COLOR_BGR2GRAY)
    img_afternoon = cv2.cvtColor(cv2.imread(afternoon.path), cv2.COLOR_BGR2GRAY)
    img_late_afternoon = cv2.cvtColor(cv2.imread(late_afternoon.path), cv2.COLOR_BGR2GRAY)

    img_late_noon_registered, _ = align.register(img_late_noon, img_noon)
    img_afternoon_registered, _ = align.register(img_afternoon, img_noon)
    img_late_afternoon_registered, _ = align.register(img_late_afternoon, img_noon)

    viz_utils.show_image('noon', img_noon)
    viz_utils.show_image('late noon', img_late_noon_registered)
    viz_utils.show_image('afternoon', img_afternoon_registered)
    viz_utils.show_image('late afternoon', img_late_afternoon_registered)
