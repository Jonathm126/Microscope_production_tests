from ximea import xiapi
import cv2
import time
import os
from ring_buffer import RingBuffer
from CTF.ctf_analyser import CTFAnalyser
avg_num_frames = 5


class XimeaSingleCamGrabber:
    def __init__(self, device_sn, params):

        cam = xiapi.Camera()
        img = xiapi.Image()

        # Open device
        print('Opening Single camera: ximea camera: ', device_sn)
        cam.open_device_by_SN(device_sn)

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

    def grab(self, show_frame_id=True):
        sample_time = time.time()
        self.cam.get_image(self.img)

        self.time_buffer.append({'time': sample_time, 'nframe': self.img.nframe})
        fps, disp_fps = self._estimate_fps()

        # create numpy array with data from camera. Dimensions of the array are
        # determined by imgdataformat
        frame = self.img.get_image_data_numpy()
        nframe = self.img.nframe

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


def save_frame(frame, out_folder, i):
    number_string = str(i).zfill(2)
    filename = "{}.png".format(number_string)
    if not os.path.exists(out_folder):
        os.makedirs(out_folder)
    output_path = os.path.join(out_folder, filename)
    cv2.imwrite(output_path, frame)
    print(f"frame {i} written to disk.")


def main(output_folder=r'calibration_data/Tracking_temp/', ctf_analyse=False):
    device_sn = 'XUCAS2213003'
    params = {'imgdataformat': 'XI_RGB24',
              'exposure': 10000,
              'gpi_selector': 'XI_GPI_PORT1',
              'gpi_mode': 'XI_GPI_TRIGGER',
              'trigger_source': 'XI_TRG_EDGE_RISING'}
    ctf_analyser = None
    if ctf_analyse:
        ctf_analyser = CTFAnalyser()

    with XimeaSingleCamGrabber(device_sn, params) as ximea_grabber:
        # out_left = FFmpegSave(width, height, path + name_left)
        # out_right = FFmpegSave(width, height, path + name_right)
        cv2.namedWindow('img', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('img', 1280, 720)
        print_usage()

        i = 1
        show_frame_id = True
        while True:
            frame = ximea_grabber.grab(show_frame_id)
            if ctf_analyse:
                # if frame is grayscale convert to color
                if len(frame.shape) == 2:
                    frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
                # detect targets
                ctf_analyser.detect_targets(frame, show=False, matching_threshold=0.7, nms_threshold=0.3, resize=0.25)
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


if __name__ == '__main__':
    output_folder = r'calibration_data/Tracking_05/'
    main(output_folder=output_folder)