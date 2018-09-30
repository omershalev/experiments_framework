import cv2

from framework import viz_utils
import computer_vision.typical_image_alignment as align

if __name__ == '__main__':

    import content.data_pointers.lavi_april_18.dji as dji_data

    obstacle = dji_data.snapshots_80_meters['15-17-1']
    clear = dji_data.snapshots_80_meters['15-10-1']

    img_obstacle = cv2.imread(obstacle.path)
    img_clear = cv2.imread(clear.path)
    img_obstacle_reg, h = align.orb_based_registration(img_obstacle, img_clear)
    # viz_utils.show_image('imreg', img_obstacle_reg)
    # img_diff = img_obstacle_reg - img_clear
    img_gray_diff = cv2.subtract(cv2.cvtColor(img_obstacle_reg, cv2.COLOR_BGR2GRAY), cv2.cvtColor(img_clear, cv2.COLOR_BGR2GRAY))
    viz_utils.show_image('imreg_diff', img_gray_diff)
    # img_diff_denoised = cv2.fastNlMeansDenoising(img_gray_diff, None, 10,10,7)
    # viz_utils.show_image('imreg_denoise', img_diff_denoised)
    # thresh = cv2.adaptiveThreshold(img_gray_diff, 0, adaptiveMethod=cv2.ADAPTIVE_THRESH_MEAN_C, thresholdType=cv2.THRESH_BINARY_INV, blockSize=21, C=2)
    # viz_utils.show_image('imreg_denoise', thresh)

    se = cv2.getStructuringElement(cv2.MORPH_RECT, (15,15))
    out = cv2.morphologyEx(img_gray_diff, cv2.MORPH_CLOSE, se)
    viz_utils.show_image('imreg_denoise', out)
