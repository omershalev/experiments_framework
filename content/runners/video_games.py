import cv2
import numpy as np
from air_ground_orchard_navigation.computer_vision import segmentation, contours_scan2
from air_ground_orchard_navigation.computer_vision.contours_scan_cython import contours_scan
from experiments_framework.framework import viz_utils

# Create a VideoCapture object and read from input file
# If the input is the camera, pass 0 instead of the video file name
cap = cv2.VideoCapture(r'/home/omer/orchards_ws/data/lavi_apr_18/raw/dji/DJI_0167.MP4')

# Check if camera opened successfully
if (cap.isOpened() == False):
    print("Error opening video stream or file")

# Read until video is completed
while (cap.isOpened()):
    # Capture frame-by-frame
    ret, frame = cap.read()
    # contours, _ = segmentation.extract_canopy_contours(frame)
    # cv2.drawContours(frame, contours, contourIdx=-1, color=(0, 255, 0), thickness=3)
    vehicle_x, vehicle_y = segmentation.extract_vehicle(frame)
    cv2.circle(frame, (vehicle_x, vehicle_y), radius=3, color=(255, 0, 255), thickness=-1)
    map_image = segmentation.extract_canopies_map(frame)
    # _, scan_coordinates_list = contours_scan2.generate(map_image, vehicle_x, vehicle_y, 0, 2 * np.pi, 360, 3, 300, 0.0125)
    import time
    ts = time.time()
    # scan_ranges = contours_scan.generate(map_image,
    #                                      center_x=vehicle_x,
    #                                      center_y=vehicle_y,
    #                                      min_angle=0,
    #                                      max_angle=2 * np.pi,
    #                                      samples_num=360,
    #                                      min_distance=3,
    #                                      max_distance=300,
    #                                      resolution=0.0125)  # TODO: fine tune parameters!
    te = time.time()
    print (te-ts)
    # for scan_coordinate in scan_coordinates_list:
    #     cv2.circle(frame, (scan_coordinate[0], scan_coordinate[1]), radius=3, color=(0, 0, 255), thickness=-1)
    if ret == True:

        # Display the resulting frame
        # cv2.imshow('Frame', frame)
        viz_utils.show_image('video', frame, wait_key=False)

        # Press Q on keyboard to  exit
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

    # Break the loop
    else:
        break

# When everything done, release the video capture object
cap.release()

# Closes all the frames
cv2.destroyAllWindows()