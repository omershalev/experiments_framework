import viz_utils
import cv2

import config

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