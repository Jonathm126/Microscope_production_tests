from ximea import xiapi
import cv2
import time
import os
from ring_buffer import RingBuffer

avg_num_frames = 5


class XimeaStereoGrabber:
    def __init__(self, devices_sn, params):
        cams = {}
        imgs = {}
        for cam_key in ['left', 'right']:
            cams[cam_key] = xiapi.Camera()
            imgs[cam_key] = xiapi.Image()

            # Open device
            print('Opening', cam_key, 'ximea camera: ', devices_sn[cam_key])
            # cams[cam_key].open_device()
            cams[cam_key].open_device_by_SN(devices_sn[cam_key])

            # Configure parameters
            for param, value in params.items():
                cams[cam_key].set_param(param, value)

        # Start acquisition
        for side in cams.keys():
            cams[side].start_acquisition()

        self.cams = cams
        self.imgs = imgs
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
        frames = []
        sample_time = time.time()
        for side in self.cams.keys():
            # get data and pass them from camera to img
            self.cams[side].get_image(self.imgs[side])

        sides = list(self.cams.keys())
        if len(sides) > 1:
            # crude synchronization
            while self.imgs[sides[-1]].nframe < self.imgs[sides[0]].nframe:
                self.cams[sides[-1]].get_image(self.imgs[sides[-1]])
            # print(self.imgs[sides[-1]].nframe - self.imgs[sides[0]].nframe)

        self.time_buffer.append({'time': sample_time, 'nframe': self.imgs[sides[0]].nframe})
        fps, disp_fps = self._estimate_fps()

        for side in self.cams.keys():
            # create numpy array with data from camera. Dimensions of the array are
            # determined by imgdataformat
            data = self.imgs[side].get_image_data_numpy()
            nframe = self.imgs[side].nframe

            if show_frame_id:
                txt = f'Frame: {nframe}, fps: {fps:.2f}, display fps: {disp_fps:.2f}'
                cv2.putText(img=data, text=txt, org=(150, 250), fontFace=cv2.FONT_HERSHEY_TRIPLEX,
                            fontScale=3,
                            color=(0, 255, 0), thickness=3)
            frames.append(data)
        return frames

    def close(self):
        for side in self.cams.keys():
            self.cams[side].stop_acquisition()
            self.cams[side].close_device()

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


def main(output_folder=r'calibration_data/Tracking_temp/'):
    devices_sn = {'left': 'XUCAS2213003', 'right': 'XUCAS2213002'}
    params = {'imgdataformat': 'XI_RGB24',
              'exposure': 10000,
              'gpi_selector': 'XI_GPI_PORT1',
              'gpi_mode': 'XI_GPI_TRIGGER',
              'trigger_source': 'XI_TRG_EDGE_RISING'}

    with XimeaStereoGrabber(devices_sn, params) as ximea_grabber:
        # out_left = FFmpegSave(width, height, path + name_left)
        # out_right = FFmpegSave(width, height, path + name_right)
        cv2.namedWindow('left', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('left', 1280, 720)
        cv2.namedWindow('right', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('right', 1280, 720)
        print_usage()

        i = 1
        show_frame_id = True
        while True:
            frames = ximea_grabber.grab(show_frame_id)
            cv2.imshow('left', frames[0])
            cv2.imshow('right', frames[1])
            key = cv2.waitKey(2) & 0xfFF
            if key == ord('q'):
                # ximea_grabber.close()
                break
            elif key == ord('c'):
                cv2.imshow('left', draw_X_on_image(frames[0]))
                cv2.imshow('right', draw_X_on_image(frames[1]))
                cv2.waitKey(5)
                for side, frame in zip(("left", "right"), frames):
                    number_string = str(i).zfill(2)
                    filename = "{}_{}.png".format(side, number_string)
                    if not os.path.exists(output_folder):
                        os.makedirs(output_folder)
                    output_path = os.path.join(output_folder, filename)
                    cv2.imwrite(output_path, frame)
                print(f"stereo pair {i} written to disk.")
                i += 1
            elif key == ord('f'):
                show_frame_id = not show_frame_id


if __name__ == '__main__':
    output_folder = r'calibration_data/Tracking_05/'
    main(output_folder=output_folder)
