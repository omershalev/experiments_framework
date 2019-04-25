import cv2
import os

from content.data_pointers.lavi_april_18 import base_resources_path, base_raw_data_path

video_path = os.path.join(base_raw_data_path, 'dji', 'DJI_0176.MP4')

if __name__ == '__main__':
    cap = cv2.VideoCapture(video_path)
    sz = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
          int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    fourcc = cv2.VideoWriter_fourcc(*'mpeg')
    out = cv2.VideoWriter()
    out.open(r'/home/omer/Documents/takeoff_output.mp4', fourcc, 30.0, sz, True)
    frame_rate = cap.get(cv2.CAP_PROP_FPS)
    frame_msec = 1000.0 / frame_rate
    frames_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    cap.set(cv2.CAP_PROP_POS_AVI_RATIO, 1)
    video_time = cap.get(cv2.CAP_PROP_POS_MSEC)
    frame_idx = frames_count - 1
    while (frame_idx > 0):
        # video_time -= frame_msec
        frame_idx -= 1
        # cap.set(cv2.CAP_PROP_POS_MSEC, video_time)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        is_success, frame = cap.read()
        if not is_success:
            continue
        out.write(frame)
        cv2.imshow('reversed', frame)
        cv2.waitKey(1)
        # if frame_idx < 750:
        #     break
    cap.release()
    out.release()
    cv2.destroyAllWindows()
