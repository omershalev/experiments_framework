import cv2
import numpy as np
import time
from skimage.measure import compare_ssim as ssim

import viz_utils


def sample_pixel_coordinates(image, multiple=False):
    class _CoordinatesSampler(object):
        def __init__(self):
            self.clicked_points = []
        def mouse_callback(self, event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN:
                if multiple:
                    self.clicked_points.append((x,y))
                else:
                    self.clicked_points = [(x,y)]

    cs = _CoordinatesSampler()
    viz_utils.show_image('image', image, wait_key=False)
    cv2.setMouseCallback('image', cs.mouse_callback)
    while True:
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
    cv2.destroyWindow('image')
    if multiple:
        return cs.clicked_points
    else:
        return cs.clicked_points[0]


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
            self.pose_time_tuples_list = []

        def mouse_callback(self, event, x, y, flags, param):
            if self.mark:
                t = time.time()
                self.pose_time_tuples_list.append((x, y, t))
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


def center_crop(image, x_ratio, y_ratio):
    x_size = image.shape[1]
    y_size = image.shape[0]
    return image[int(y_ratio * y_size):int((1 - y_ratio) * y_size), int(x_ratio * x_size):int((1 - x_ratio) * x_size)]


def crop_region(image, x_center, y_center, x_pixels, y_pixels):
    x_size = image.shape[1]
    y_size = image.shape[0]
    image = image[max(0, y_center - y_pixels / 2) : min(y_size, y_center + y_pixels / 2),
                  max(0, x_center - x_pixels / 2) : min(x_size, x_center + x_pixels / 2)]
    upper_left = (max(0, x_center - x_pixels / 2), max(0, y_center - y_pixels / 2))
    lower_right = (min(x_size, x_center + x_pixels / 2), min(y_size, y_center + y_pixels / 2))
    return image, upper_left, lower_right


def insert_image_patch(image, patch, upper_left, lower_right):
    image[upper_left[1] : lower_right[1], upper_left[0] : lower_right[0]] = patch
    return image


def get_coordinates_list_from_scan_ranges(scan_ranges, center_x, center_y, min_angle, max_angle, resolution):
    samples_num = len(scan_ranges)
    coordinates_list = []
    for scan_idx, theta in enumerate(np.linspace(min_angle, max_angle, num=samples_num)):
        if np.isnan(scan_ranges[scan_idx]):
            continue
        x = int(np.round(center_x + scan_ranges[scan_idx] / resolution * np.cos(-theta)))
        y = int(np.round(center_y + scan_ranges[scan_idx] / resolution * np.sin(-theta)))
        coordinates_list.append((x,y))
    return coordinates_list


def warp_image(image, points_in_image, points_in_baseline, method='homographic'):
    if method == 'homographic':
        h, _ = cv2.findHomography(np.float32(points_in_image), np.float32(points_in_baseline))
        warpped_image = cv2.warpPerspective(image, h, (image.shape[1], image.shape[0]))
    elif method == 'affine':
        M, _ = cv2.estimateAffine2D(np.float32(points_in_image), np.float32(points_in_baseline))
        warpped_image = cv2.warpAffine(image, M, (image.shape[1], image.shape[0]))
    elif method == 'rigid':
        M = cv2.estimateRigidTransform(np.float32(points_in_image), np.float32(points_in_baseline), fullAffine=False)
        warpped_image = cv2.warpAffine(image, M, (image.shape[1], image.shape[0]))
    else:
        raise Exception('Unsupported method')
    return warpped_image


def calculate_image_diff(image1, image2, method='mse', x_crop_ratio=0.3, y_crop_ratio=0.3):
    if image1.shape != image2.shape:
        raise Exception('Two images must have the same dimension')
    image1 = center_crop(image1, x_crop_ratio, y_crop_ratio)
    image2 = center_crop(image2, x_crop_ratio, y_crop_ratio)
    if method == 'mse':
        err = np.sum((image1.astype('float') - image2.astype('float')) ** 2)
        err /= float(image1.shape[0] * image1.shape[1])
    elif method == 'ssim':
        err = ssim(image1, image2)
    else:
        raise Exception('Unsupported method')
    return err