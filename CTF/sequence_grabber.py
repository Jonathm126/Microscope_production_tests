# from ximea.ximea_single_cam_grabber import XimeaSingleCamGrabber
import cv2
import time
import os
# from ximea.ring_buffer import RingBuffer
import sys
import tkinter as tk
from tkinter import filedialog, simpledialog, messagebox

sys.path.append("../")
from CTF.ctf_analyser import CTFAnalyser
# from CTF.Thorlabs_z_stage.thorlabs_kinesis import ThorlabsZStage
import json
import numpy as np
from argparse import ArgumentParser
import pandas as pd
import matplotlib.pyplot as plt


class SequenceRunner:
    def __init__(self, motion_json_filemname='Thorlabs_z_stage/motion_parameters.json'):
        with open(motion_json_filemname) as f:
            motion_params = json.load(f)
        self.motion_params = motion_params
        self.data_list = []
        self.z_waypoints = np.arange(motion_params['min'],
                                     motion_params['max'],
                                     motion_params['step'])
        self.wait_time = motion_params['freeze_time_sec']
        self.z_waypoints_rel = np.insert(np.diff(self.z_waypoints), 0, motion_params['min'])
        self.ind = 0

    def move_to_next_waypoint(self, stage):
        if self.ind < len(self.z_waypoints):
            stage.move_to(self.z_waypoints_rel[self.ind], rel=True)
            time.sleep(self.wait_time)
            self.ind += 1
            return True
        else:
            print("End of sequence reached.")
            return False

    def append_observation(self, z, iris_mm, ctf_data):
        data = {'z': z, 'iris_mm': iris_mm, 'ctf_data': ctf_data}
        self.data_list.append(data)

    def save(self, filename):
        df = pd.DataFrame(self.data_list)
        df.to_csv(filename, index=False)
        print(f"Sequence data saved to {filename}")

    def load(self, filename):
        df = pd.read_csv(filename)
        self.data_list = df.to_dict(orient='records')

    def plot(self, export_pdf=False, output_folder=None):
        num_waypoints = len(self.data_list)
        ctf_data_list = [d['ctf_data'] for d in self.data_list]
        z = [d['z'] for d in self.data_list]
        iris_mm = [d['iris_mm'] for d in self.data_list]
        freqs = ctf_data_list[0]['subbox'].values()[:4]

        for target_ind in range(ctf_data_list[0]['target'][-1]):
            # show ctf data vs z values for each targets groups in CTF data
            # on each graph we will have 4 plots correspond to targets subboxes
            plt.figure()
            plt.title(f"CTF vs Z for target {target_ind}, iris_mm: {iris_mm[0]}")
            plt.xlabel("Z [mm]")
            plt.ylabel("CTF [%]")

            for i, freq in enumerate(freqs):
                curr_ctf = []
                for j in range(num_waypoints):
                    data = ctf_data_list[j]
                    if data['subbox'] == freq and data['target'] == target_ind:
                        curr_ctf.append(data['ctf'])
                plt.plot(z, curr_ctf, '-o', label=f"freq: {freq}")
            plt.legend()
            plt.grid(True)
            plt.show()
            if export_pdf:
                if output_folder:
                    plt.savefig(os.path.join(output_folder, f"CTF_vs_Z_target_{target_ind}_iris_mm_{iris_mm[0]}.pdf"))
                else:
                    print("Output folder not specified, cannot save pdfs.")


def print_usage():
    print('Sequence CTF analyser:')
    print('\tq: quit')
    print('\tc: record frame')
    print('\tf: toggle display frame number')
    print('\tz: zoom on targets')


def save_image(frame, output_folder, cnt, z_waypoints, iris_mm, sufix=""):
    number_string = str(cnt).zfill(2)
    filename = "img-{}_z-{}_iris-mm-{}{}.png".format(number_string, z_waypoints[cnt], iris_mm, sufix)
    output_path = os.path.join(output_folder, filename)
    print(f"frame {sufix} {cnt} written to disk: {output_path}")
    cv2.imwrite(output_path, frame)


def main(args):
    timestr = time.strftime("%Y-%m-%d_%H-%M-%S")
    output_folder = os.path.join(args.output_folder, timestr + '_iris-mm-' + args.iris_mm)
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Opening cameras cam_params JSON file
    cam_json_filemname = 'ximea/left_mic_params.json'
    with open(cam_json_filemname) as f:
        cam_params = json.load(f)

    # initialize ctf analyser
    zoom = False
    ctf_analyser = CTFAnalyser(template_path="../CTF/template/template.bmp")

    # initialize thorlabs motion controller
    stage = ThorlabsZStage()

    # initialize sequence runner
    sequence_runner = SequenceRunner()

    with XimeaSingleCamGrabber(cam_params) as ximea_grabber:
        cv2.namedWindow('img', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('img', 2056, 1504)
        print_usage()

        cnt = 0
        show_frame_id = True
        while True:
            frame = ximea_grabber.grab(show_frame_id, rotate_180=args.rotate_180)

            # if frame is in RAW8 convert to BGR
            if 'RAW8' in cam_params['imgdataformat']:
                frame = cv2.cvtColor(frame, cv2.COLOR_BAYER_RG2BGR)
            elif 'MONO8' in cam_params['imgdataformat']:
                frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

            # move to next waypoint
            if not sequence_runner.move_to_next_waypoint(stage):
                break

            # save raw image to disk
            save_image(frame, output_folder, cnt, sequence_runner.z_waypoints, args.iris_mm, sufix="_raw")

            # detect targets
            ctf_analyser.detect_targets(frame, matching_threshold=0.7, nms_threshold=0.3, resize=0.25)
            if zoom:
                frame_zoom = ctf_analyser.get_boxes_zoomed(frame, expand=100)

            # save detected image to disk
            save_image(frame, output_folder, cnt, sequence_runner.z_waypoints, args.iris_mm, sufix='_detected')

            # append observation to sequence runner
            sequence_runner.append_observation(sequence_runner.z_waypoints[cnt], args.iris_mm, ctf_analyser.data)

            # clear ctf object
            ctf_analyser.clear_data()

            cnt += 1

            if zoom:
                cv2.imshow('img', frame_zoom)
            else:
                cv2.imshow('img', frame)
            key = cv2.waitKey(2) & 0xfFF
            if key == ord('q'):
                break
            elif key == ord('f'):
                show_frame_id = not show_frame_id
            elif key == ord('z'):
                zoom = not zoom

        # save data
        sequence_runner.save(os.path.join(output_folder, 'sequence_data.csv'))

        # plot data
        if args.plot_at_end:
            sequence_runner.plot(export_pdf=True)

        # bring stage back to home
        stage.move_to(-sequence_runner.z_waypoints[-1], rel=True)
        stage.close()


class Args:
    def __init__(self):
        self.output_folder = '/log'
        self.iris_mm = 12
        self.rotate_180 = True
        self.plot_at_end = True


if __name__ == '__main__':
    # Initialize Args instance
    args = Args()

    # Create the root window
    root = tk.Tk()
    root.withdraw()  # We don't want a full GUI, so keep the root window from appearing for now

    # Show opening message box
    messagebox.showinfo("CTF sequence grabber","Welcome to CTF sequence grabber, Make sure: \n1. target is in focus plain\n2. Iris is set and known. \n3. LED is ON. \nPress OK to continue.")

    # Ask for the output folder
    args.output_folder = filedialog.askdirectory(title="Select Output Folder")
    if not args.output_folder:
        messagebox.showerror("Error", "Output folder is required!")
        exit(1)

    # Ask for iris configuration
    iris_mm_input = simpledialog.askstring("Input", "Enter iris configuration in mm:")
    if iris_mm_input:  # Only update if the user provided a value
        args.iris_mm = iris_mm_input

    # Ask for rotate_180
    args.rotate_180 = messagebox.askyesno("Input", "Rotate the images 180 degrees?")

    # Ask for plot_at_end
    args.plot_at_end = messagebox.askyesno("Input", "Plot graphs at the end?")

    # Call the main function with the collected arguments
    main(args)
