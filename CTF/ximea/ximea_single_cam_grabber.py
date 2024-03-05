from ximea import xiapi
import cv2
import time
import os
from ring_buffer import RingBuffer
import sys
sys.path.append("../")
sys.path.append("../../")
from CTF.ctf_analyser import CTFAnalyser
import json

avg_num_frames = 5


class XimeaSingleCamGrabber:
    def __init__(self, params):

        cam = xiapi.Camera()
        img = xiapi.Image()

        # Open device
        print('Opening Single camera: ximea camera: ', params['device_sn'])
        cam.open_device_by_SN(params['device_sn'])
        del params['device_sn']

        # Configure parameters
        for param, value in params.items():
            cam.set_param(param, value)
        cam.start_acquisition()

        self.cam = cam
        self.img = img
        self.time_buffer = RingBuffer(avg_num_frames)

    def _estimate_fps(self):
        time_vec = self.time_buffer.get()
        if len(time_vec) < 2:
            return 0, 0
        dt = time_vec[-1]['time'] - time_vec[0]['time']
        n = time_vec[-1]['nframe'] - time_vec[0]['nframe'] + 1
        fps = n / dt
        disp_fps = len(time_vec) / dt
        return fps, disp_fps

    def grab(self, show_frame_id=True, rotate_180=False):
        sample_time = time.time()
        self.cam.get_image(self.img)

        self.time_buffer.append({'time': sample_time, 'nframe': self.img.nframe})
        fps, disp_fps = self._estimate_fps()

        # create numpy array with data from camera. Dimensions of the array are
        # determined by imgdataformat
        frame = self.img.get_image_data_numpy()
        nframe = self.img.nframe
        # rotate image
        if rotate_180:
            frame = cv2.rotate(frame, cv2.ROTATE_180)

        if show_frame_id:
            txt = f'Frame: {nframe}, fps: {fps:.2f}, display fps: {disp_fps:.2f}'
            cv2.putText(img=frame, text=txt, org=(150, 250), fontFace=cv2.FONT_HERSHEY_TRIPLEX,
                        fontScale=3,
                        color=(0, 255, 0), thickness=3)
        return frame

    def close(self):
        self.cam.stop_acquisition()
        self.cam.close_device()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


def draw_X_on_image(frame):
    color = (0, 0, 255)
    thickness = 3

    sy, sx, sz = frame.shape
    frame2 = frame.copy()
    frame2 = cv2.line(frame2, (0, 0), (sx, sy), color, thickness)
    frame2 = cv2.line(frame2, (0, sy), (sx, 0), color, thickness)
    return frame2


def print_usage():
    print('Ximea grabber:')
    print('\tq: quit')
    print('\tc: record frame')
    print('\tf: toggle display frame number')
    print('\tz: zoom on targets')


def save_frame(frame, out_folder, i):
    number_string = str(i).zfill(2)
    filename = "{}.png".format(number_string)
    if not os.path.exists(out_folder):
        os.makedirs(out_folder)
    output_path = os.path.join(out_folder, filename)
    cv2.imwrite(output_path, frame)
    print(f"frame {i} written to disk.")


def main(output_folder=r'calibration_data/Tracking_temp/', ctf_analyse=True, rotate_180=True):
    # Opening JSON file
    cam_json_filemname = 'left_mic_params.json'
    with open(cam_json_filemname) as f:
        params = json.load(f)

    ctf_analyser = None
    zoom = False
    if ctf_analyse:
        # ctf_analyser = CTFAnalyser()
        ctf_analyser = CTFAnalyser(template_path="../../CTF/template/template.bmp")

    with XimeaSingleCamGrabber(params) as ximea_grabber:
        # out_left = FFmpegSave(width, height, path + name_left)
        # out_right = FFmpegSave(width, height, path + name_right)
        cv2.namedWindow('img', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('img', 2056, 1504)
        print_usage()

        i = 1
        show_frame_id = True
        while True:
            frame = ximea_grabber.grab(show_frame_id, rotate_180=rotate_180)
            if ctf_analyse:
                # if frame is grayscale convert to color
                if len(frame.shape) == 2:
                    frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
                # detect targets
                ctf_analyser.detect_targets(frame, show=False, matching_threshold=0.7, nms_threshold=0.3, resize=0.25)
                if zoom:
                    frame_zoom = ctf_analyser.get_boxes_zoomed(frame, expand=100)
            if zoom:
                cv2.imshow('img', frame_zoom)
            else:
                cv2.imshow('img', frame)
            key = cv2.waitKey(2) & 0xfFF
            if key == ord('q'):
                break
            elif key == ord('c'):
                cv2.imshow('img', draw_X_on_image(frame))
                cv2.waitKey(5)
                save_frame(frame, output_folder, i)
                i += 1
            elif key == ord('f'):
                show_frame_id = not show_frame_id
            elif key == ord('z'):
                zoom = not zoom


if __name__ == '__main__':
    output_folder = r'calibration_data/Tracking_05/'
    main(output_folder=output_folder)