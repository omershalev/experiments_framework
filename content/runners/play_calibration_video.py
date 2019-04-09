from framework import ros_utils

if __name__ == '__main__':
    ros_utils.play_video_to_topic(r'/home/omer/orchards_ws/resources/dji_calibration/DJI_0209.MP4', topic='/camera/image_raw', frame_id='camera_link')