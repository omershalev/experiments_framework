import cv2
import numpy as np
import datetime

import viz_utils


def sample_pixel_coordinates(image):
    class _CoordinatesSampler(object):
        def __init__(self):
            self.sampled_x = None
            self.sampled_y = None
        def mouse_callback(self, event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN:
                self.sampled_x, self.sampled_y = x, y

    cs = _CoordinatesSampler()
    viz_utils.show_image('image', image, wait_key=False)
    cv2.setMouseCallback('image', cs.mouse_callback)
    while True:
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
    return cs.sampled_x, cs.sampled_y


def sample_hsv_color(image):
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    sampled_pixels = []
    def mouse_callback(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            pixel_hsv = hsv_image[(y,x)]
            print (str(pixel_hsv))
            sampled_pixels.append(pixel_hsv)
    viz_utils.show_image('image', image, wait_key=False)
    cv2.setMouseCallback('image', mouse_callback)
    while True:
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
    return sampled_pixels



def mark_trajectory_on_image(image):
    class _TrajectoryMarker(object):
        def __init__(self, image):
            self.mark = False
            if len(np.shape(image)) == 2:
                self.image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
            else:
                self.image = image
            self.image = image
            self.pose_time_tuples_list = []

        def mouse_callback(self, event, x, y, flags, param):
            if self.mark:
                self.pose_time_tuples_list.append((x, y, datetime.datetime.now()))
                for (x, y) in self.pose_time_tuples_list:
                    self.image = cv2.circle(img=self.image, center=(x, y), radius=2, color=(255, 255, 255),
                                            thickness=-1)
                viz_utils.show_image('image', self.image, wait_key=False)
            if event == cv2.EVENT_LBUTTONDOWN:
                self.mark = True
            if event == cv2.EVENT_LBUTTONUP:
                self.mark = False

    tm = _TrajectoryMarker(image)
    viz_utils.show_image('image', image, wait_key=False)
    cv2.setMouseCallback('image', tm.mouse_callback)
    while True:
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
    return tm.pose_time_tuples_list