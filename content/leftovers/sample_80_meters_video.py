from framework import cv_utils
from framework import utils

video_path = r'/home/omer/orchards_ws/resources/lavi_apr_18/raw/dji/DJI_0168.MP4'

if __name__ == '__main__':
    execution_dir = utils.create_new_execution_folder('80_meters_video_samples')
    cv_utils.sample_video(video_path, output_path=execution_dir, start_time=22, stop_time=24, samples=10)