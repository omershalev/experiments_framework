import cv2
import numpy as np
import datetime
import time

import viz_utils


def sample_pixel_coordinates(image):
    class _CoordinatesSampler(object):
        def __init__(self):
            self.clicked_x = None
            self.clicked_y = None
        def mouse_callback(self, event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN:
                self.clicked_x, self.clicked_y = x, y

    cs = _CoordinatesSampler()
    viz_utils.show_image('image', image, wait_key=False)
    cv2.setMouseCallback('image', cs.mouse_callback)
    while True:
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
    cv2.destroyWindow('image')
    return cs.clicked_x, cs.clicked_y


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
                # self.pose_time_tuples_list.append((x, y, datetime.datetime.now())) # TODO: remove
                self.pose_time_tuples_list.append((x, y, time.time()))
                for (x, y, _) in self.pose_time_tuples_list:
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


def mark_rectangle_on_image(image, points):
    image_with_rectangle = image.copy()
    for points_pair in zip(points[-1:] + points[:-1], points):
        image_with_rectangle = cv2.line(image_with_rectangle, tuple(points_pair[0]), tuple(points_pair[1]), color=(0, 255, 0), thickness=2)
    return image_with_rectangle


def get_bounding_box(image, points, expand_ratio=0.0):
    x, y, w, h = cv2.boundingRect(np.array(points))
    upper_left_x = max(0, x - int(expand_ratio*w))
    upper_left_y = max(0, y - int(expand_ratio*h))
    lower_right_x = min(image.shape[1], x + int((1+expand_ratio)*w))
    lower_right_y = min(image.shape[0], y + int((1+expand_ratio)*h))
    return (upper_left_x, upper_left_y), (lower_right_x, lower_right_y)


def mark_bounding_box(image, points, expand_ratio=0.0):
    (upper_left_x, upper_left_y), (lower_right_x, lower_right_y) = get_bounding_box(image, points, expand_ratio)
    cv2.rectangle(image, (upper_left_x, upper_left_y), (lower_right_x, lower_right_y), (255, 255, 0), 2)