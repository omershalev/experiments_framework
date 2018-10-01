import os
import colorsys
import numpy as np
import cv2


import experiments_framework.framework.viz_utils as viz_utils








# def alignImages(im1, im2):
#     MAX_FEATURES = 500
#     GOOD_MATCH_PERCENT = 0.15
#
#     # Convert images to grayscale
#     im1Gray = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
#     im2Gray = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)
#
#     # Detect ORB features and compute descriptors.
#     orb = cv2.ORB_create(MAX_FEATURES)
#     keypoints1, descriptors1 = orb.detectAndCompute(im1Gray, None)
#     keypoints2, descriptors2 = orb.detectAndCompute(im2Gray, None)
#
#     # Match features.
#     matcher = cv2.DescriptorMatcher_create(cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING)
#     matches = matcher.match(descriptors1, descriptors2, None)
#
#     # Sort matches by score
#     matches.sort(key=lambda x: x.distance, reverse=False)
#
#     # Remove not so good matches
#     numGoodMatches = int(len(matches) * GOOD_MATCH_PERCENT)
#     matches = matches[:numGoodMatches]
#
#     # Draw top matches
#     imMatches = cv2.drawMatches(im1, keypoints1, im2, keypoints2, matches, None)
#     viz_utils.show_image('matches', imMatches)
#
#     # Extract location of good matches
#     points1 = np.zeros((len(matches), 2), dtype=np.float32)
#     points2 = np.zeros((len(matches), 2), dtype=np.float32)
#
#     for i, match in enumerate(matches):
#         points1[i, :] = keypoints1[match.queryIdx].pt
#         points2[i, :] = keypoints2[match.trainIdx].pt
#
#     # Find homography
#     h, mask = cv2.findHomography(points1, points2, cv2.RANSAC)
#
#     # Use homography
#     height, width, channels = im2.shape
#     im1Reg = cv2.warpPerspective(im1, h, (width, height))
#
#     return im1Reg, h



if __name__ == '__main__':

    ############################ Draw Contours ############################
    green_lower_hue_degrees = 65 # was: 65
    green_lower_saturation_percent = 5
    green_lower_value_percent = 0
    green_upper_hue_degrees = 160 # was: 165
    green_upper_saturation_percent = 100
    green_upper_value_percent = 100

    # azure_lower_hue_degrees = 180
    # azure_lower_saturation_percent = 5
    # azure_lower_value_percent = 0
    # azure_upper_hue_degrees = 220
    # azure_upper_saturation_percent = 100
    # azure_upper_value_percent = 100

    lower_color = np.array([green_lower_hue_degrees / 2, green_lower_saturation_percent * 255 / 100, green_lower_value_percent * 255 / 100])
    upper_color = np.array([green_upper_hue_degrees / 2, green_upper_saturation_percent * 255 / 100, green_upper_value_percent * 255 / 100])
    #
    # lower_color = np.array([azure_lower_hue_degrees / 2, azure_lower_saturation_percent * 255 / 100, azure_lower_value_percent * 255 / 100])
    # upper_color = np.array([azure_upper_hue_degrees / 2, azure_upper_saturation_percent * 255 / 100, azureeu_upper_value_percent * 255 / 100])
    #

    import experiments_framework.content.data_pointers.lavi_april_18.dji as dji_data

    for snapshot_name, snapshot_descriptor in dji_data.snapshots_60_meters.items() + dji_data.snapshots_80_meters.items():
        img = cv2.imread(snapshot_descriptor.path)
        draw_contours(img, lower_color, upper_color)


