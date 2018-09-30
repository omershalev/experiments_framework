import cv2

import experiments_framework.framework.viz_utils as viz_utils
import air_ground_orchard_navigation.computer_vision.image_alignment as image_alignment
import air_ground_orchard_navigation.computer_vision.segmentation as canopy_contours

if __name__ == '__main__':

    import experiments_framework.content.data_pointers.lavi_april_18.dji as dji_data

    noon = dji_data.snapshots_80_meters['15-08-1']
    late_noon = dji_data.snapshots_80_meters['15-53-1']
    afternoon = dji_data.snapshots_80_meters['16-55-1']
    late_afternoon = dji_data.snapshots_80_meters['19-04-1']

    img_noon = cv2.imread(noon.path)
    img_late_noon = cv2.imread(late_noon.path)
    img_afternoon = cv2.imread(afternoon.path)
    img_late_afternoon = cv2.imread(late_afternoon.path)

    _, noon_canopy_contours = canopy_contours.extract_canopy_contours(img_noon)
    _, late_noon_canopy_contours = canopy_contours.extract_canopy_contours(img_late_noon)
    _, afternoon_canopy_contours = canopy_contours.extract_canopy_contours(img_afternoon)
    _, late_afternoon_canopy_contours = canopy_contours.extract_canopy_contours(img_late_afternoon)

    i1, h1 = image_alignment.typical_registration(late_noon_canopy_contours, noon_canopy_contours, show_matches=True)
    i2, h2 = image_alignment.typical_registration(afternoon_canopy_contours, noon_canopy_contours, show_matches=True)
    i3, h3 = image_alignment.typical_registration(late_afternoon_canopy_contours, noon_canopy_contours, show_matches=True)

    height, width = img_noon.shape[0], img_noon.shape[1]
    img_late_noon_registered = cv2.warpPerspective(img_late_noon, h1, (width, height))
    img_afternoon_registered = cv2.warpPerspective(img_afternoon, h2, (width, height))
    img_late_afternoon_registered = cv2.warpPerspective(img_late_afternoon, h3, (width, height))
    # img_late_noon_registered = cv2.warpAffine(img_late_noon, h1, (width, height))
    # img_afternoon_registered = cv2.warpAffine(img_afternoon, h2, (width, height))
    # img_late_afternoon_registered = cv2.warpAffine(img_late_afternoon, h3, (width, height))


    viz_utils.show_image('noon', img_noon)
    viz_utils.show_image('late noon', img_late_noon_registered)
    viz_utils.show_image('afternoon', img_afternoon_registered)
    viz_utils.show_image('late afternoon', img_late_afternoon_registered)
    # viz_utils.show_image('late noon', i1)
    # viz_utils.show_image('afternoon', i2)
    # viz_utils.show_image('late afternoon', i3)
