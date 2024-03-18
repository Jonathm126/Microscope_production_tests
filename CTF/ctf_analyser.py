import os
import cv2
import numpy as np
from CTF.utils import crop_frame_bbox, non_max_suppression_fast
from scipy.signal import find_peaks
import matplotlib.pyplot as plt
from scipy.fft import fft
import time
import cProfile
import pstats
from matplotlib.backends.backend_pdf import PdfPages
import pandas as pd


class CTFAnalyser:
    def __init__(self, template_path='template/template.bmp', resolution=(4112, 3008), margin=5,
                 export_pdf_filename=None):
        self.window = 'CTF_window'
        cv2.namedWindow(self.window, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.window, 1280, 720)
        self.template = cv2.imread(template_path)
        self.width, self.height = resolution
        self.bboxes = []
        self.margin = margin
        self.rel_subboxes = {'13um_h': [223, 103, 268, 148],
                             '13um_v': [161, 103, 204, 148],
                             '27um_h': [74, 103, 117, 148],
                             '27um_v': [11, 103, 54, 148],
                             '54um_h': [74, 15, 117, 60],
                             '54um_v': [11, 15, 54, 60],
                             '108um_h': [223, 15, 268, 60],
                             '108um_v': [161, 15, 204, 60]}
        for key, value in self.rel_subboxes.items():
            value[0] += self.margin
            value[1] += self.margin
            value[2] -= self.margin
            value[3] -= self.margin
        self.sorted_mapping = np.array([10, 11, 12, 8, 9, 7, 2, 1, 6, 3, 5, 4, 13, 14, 15]) - 1
        self.export_pdf_filename = export_pdf_filename
        self.data = []
        if export_pdf_filename:
            plt.ion()
            plt.clf()
            self.pdf_file = PdfPages(export_pdf_filename)
            self.fig = plt.figure(figsize=(15, 10))

    def detect_targets(self, main_image, show=True, matching_threshold=0.7, nms_threshold=0.3, resize=None):
        # Load the main image and template
        h, w = self.template.shape[:2]
        # Convert images to grayscale
        main_image_gray = cv2.cvtColor(main_image, cv2.COLOR_BGR2GRAY)
        main_image_gray_raw = main_image_gray.copy()
        template_gray = cv2.cvtColor(self.template, cv2.COLOR_BGR2GRAY)

        # Perform template matching
        if resize is not None:
            main_image_gray = cv2.resize(main_image_gray, (
            int(resize * main_image_gray.shape[1]), int(resize * main_image_gray.shape[0])))
            template_gray = cv2.resize(template_gray,
                                       (int(resize * template_gray.shape[1]), int(resize * template_gray.shape[0])))
        res = cv2.matchTemplate(main_image_gray, template_gray, cv2.TM_CCOEFF_NORMED)
        loc = np.where(res >= matching_threshold)

        # Create a list of bounding boxes
        boxes = []
        for pt in zip(*loc[::-1]):
            if resize is not None:
                boxes.append([pt[0] // resize, pt[1] // resize, pt[0] // resize + w, pt[1] // resize + h])
            else:
                boxes.append([pt[0], pt[1], pt[0] + w, pt[1] + h])

        boxes = np.array(boxes)

        # Apply non-maximum suppression
        nms_boxes = non_max_suppression_fast(boxes, nms_threshold)
        nms_boxes_arranged = self.sort_targets(nms_boxes)
        self.bboxes = nms_boxes_arranged

        # Draw rectangles around the matches
        for i, (startX, startY, endX, endY) in enumerate(nms_boxes_arranged):
            cv2.rectangle(main_image, (startX, startY), (endX, endY), (0, 255, 0), 2)
            cv2.putText(main_image, str(i + 1), (startX, startY), cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 255, 0), 2)
            cv2.drawMarker(main_image, (int((startX + endX) / 2), int((startY + endY) / 2)), (255, 0, 255),
                           markerType=cv2.MARKER_CROSS, markerSize=20, thickness=2)
            # draw rel subboxes
            for key, value in self.rel_subboxes.items():
                x1, y1, x2, y2 = int(startX + value[0]), int(startY + value[1]), int(startX + value[2]), int(
                    startY + value[3])
                cv2.rectangle(main_image, (x1, y1), (x2, y2),
                              (255, 0, 0), 2)
                cv2.putText(main_image, key, (x1, y1),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 255), 1)
                # ctf, _, _ = self.calculate_and_visualize_ctf(main_image_gray, [x1, y1, x2, y2], show=False)
                ctf, mtf, mtf_max_f = self.calculate_and_visualize_mtf(main_image_gray_raw, [x1, y1, x2, y2], key[-1], show=show,
                                                            text=f'target: {i + 1}, {key}')
                curr_data = {'target': i + 1, 'subbox': key, 'ctf': ctf, 'mtf': mtf, 'mtf_max_f': mtf_max_f}
                self.data.append(curr_data)
                if ctf is not None:
                    cv2.putText(main_image, f'{int(ctf)} %', ((x1 + x2) // 2, (y1 + y2) // 2),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        # Display the result
        if show:
            cv2.imshow(self.window, main_image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

    def sort_targets(self, nms_boxes):
        if len(nms_boxes) != 15:
            print('Not all targets were detected!')
            return nms_boxes
        boxes_percentage = nms_boxes / np.array([self.width, self.height, self.width, self.height]) * 10
        bbox_centers = (boxes_percentage[:, :2] + boxes_percentage[:, 2:]) // 2
        sorted_inds = np.lexsort((bbox_centers[:, 1], bbox_centers[:, 0]))
        nms_boxes_arranged = np.zeros_like(nms_boxes)
        nms_boxes_arranged[self.sorted_mapping] = nms_boxes[sorted_inds]
        return nms_boxes_arranged

    @staticmethod
    def calculate_and_visualize_ctf(image, bbox, show=True):
        # Extract the ROI from the image using the x1, y1, x2, y2 format
        x1, y1, x2, y2 = bbox
        roi = image[y1:y2, x1:x2]

        # Compute the histogram of the ROI
        hist, bin_edges = np.histogram(roi, bins=256, range=(0, 255))

        # Find peaks in the histogram
        peaks, _ = find_peaks(hist, prominence=10)
        if show:
            # Visualize the histogram and peaks
            plt.figure(figsize=(10, 6))
            plt.plot(hist, label='Histogram')
            plt.plot(peaks, hist[peaks], "x", label='Detected Peaks')
            plt.title('Histogram and Detected Peaks')
            plt.xlabel('Pixel Intensity')
            plt.ylabel('Frequency')
            plt.legend()
            plt.show()

        # Handle special case and calculate CTF
        if len(peaks) == 1 and bin_edges[peaks[0]] in range(100, 156):  # Assuming mid-grey is between 100 and 155
            return 0, peaks[0], peaks[0]  # Only one peak at mid-grey, return 0 CTF

        elif len(peaks) >= 2:
            sorted_peaks = sorted(peaks, key=lambda x: hist[x], reverse=True)
            white_peak, black_peak = sorted_peaks[:2]
            white_peak, black_peak = max(white_peak, black_peak), min(white_peak, black_peak)
            ctf = np.abs(hist[white_peak] - hist[black_peak]) / (hist[white_peak] + hist[black_peak]) * 100
            return ctf, white_peak, black_peak
        else:
            return None, None, None  # Not enough peaks found

    def calculate_and_visualize_mtf(self, image, bbox, direction, show=True, text=None, peak_threshold=2):
        # Crop the image to the region of interest
        x1, y1, x2, y2 = bbox
        roi = image[y1:y2, x1:x2]

        # Convert to grayscale if necessary
        if len(roi.shape) == 3:
            roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

        # Calculate the Edge Spread Function (ESF)
        if direction == 'h':
            esf = np.mean(roi, axis=1)
        elif direction == 'v':
            esf = np.mean(roi, axis=0)
        else:
            raise ValueError("Direction must be 'h' or 'v'")

        # Find peaks in the ESF
        peaks_up, _ = find_peaks(esf, threshold=peak_threshold)
        peaks_down, _ = find_peaks(-esf, threshold=peak_threshold)
        peaks = np.concatenate((peaks_up, peaks_down))

        # calculate CTF
        if peaks_up.size == 0 or peaks_down.size == 0:
            ctf = 0
        else:
            esf_up = np.mean(esf[peaks_up])
            esf_down = np.mean(esf[peaks_down])
            ctf = (esf_up - esf_down) / (esf_up + esf_down) * 100  # * np.pi /4

        # Calculate the Line Spread Function (LSF) as the derivative of the ESF
        lsf = np.diff(esf)
        # lsf_groups = np.split(lsf, peaks)

        # Calculate the Modulation Transfer Function (MTF) as the Fourier Transform of the LSF
        # mtfs = []
        # for i, lsf in enumerate(lsf_groups):
        #     if i % 2 == 0:
        #         lsf = -lsf
        #     if len(lsf) == 0:
        #         continue
        #     mtf = np.abs(fft(lsf))
        #     mtf = mtf[:len(mtf) // 2]  # Use only the first half which is symmetric
        #     mtfs.append(mtf)
        #
        # mtf = np.mean(mtfs, axis=0)

        if len(lsf) == 0:
            return ctf, None, None

        mtf = np.abs(fft(lsf))
        mtf = mtf[:len(mtf) // 2]  # Use only the first half which is symmetric

        mtf_raw_max = np.max(mtf)
        max_mtf_freq = np.argmax(mtf)

        # Normalize MTF by its maximum value
        mtf_normalized = mtf / np.max(mtf)

        # Visualize MTF if show is True
        if show:
            if export_pdf_filename is None:
                fig = plt.figure(figsize=(15, 10))
            else:
                fig = self.fig
            fig.suptitle(f'MTF analysis: {text}')
            ax1 = plt.subplot(121)
            ax1.imshow(roi, cmap='gray')
            ax2 = plt.subplot(322)
            ax2.plot(esf, label='ESF')
            ax2.plot(peaks, esf[peaks], "x", label='Detected Peaks')
            ax2.set_xlabel('pixels')
            ax2.set_ylabel('gray scal values')
            ax2.set_title(f'Edge Spread Function (ESF):')
            ax2.legend()
            ax2.grid(True)

            ax3 = plt.subplot(324)
            ax3.plot(lsf, label='LSF')
            ax3.set_xlabel('pixels')
            ax3.set_ylabel('gray scal derivative values')
            ax3.set_title(f'Line Spread Function (LSF):')
            ax3.legend()
            ax3.grid(True)

            ax4 = plt.subplot(326)
            ax4.plot(mtf_normalized, label='MTF_narmalized')
            # ax4.plot(mtf, label='MTF')
            ax4.set_xlabel('Frequency (cycles/pixel)')
            ax4.set_ylabel('Contrast')
            ax4.set_title('Modulation Transfer Function (MTF):')
            ax4.legend()
            ax4.grid(True)
            fig.tight_layout(pad=2)
            plt.show()
            if self.export_pdf_filename is not None:
                plt.pause(0.00001)
                fig.savefig(self.pdf_file, format='pdf')
                fig.clear()
        return ctf, mtf_raw_max, max_mtf_freq

    def get_boxes_zoomed(self, frame, expand=100):
        if self.bboxes.shape[0] != 15:
            print('not all targets are visible, aborting zoom')
            return frame

        h, w = self.template.shape[:2]
        if np.all(w == np.abs(self.bboxes[:, 0] - self.bboxes[:, 2])) and np.all(h == np.abs(self.bboxes[:, 1] - self.bboxes[:, 3])):

            out_frame = np.zeros((3*h, 5*w, 3), dtype=np.uint8)

            out_frame[:h, :w] = crop_frame_bbox(frame, self.bboxes[10-1], str(10))
            out_frame[h:2*h, :w] = crop_frame_bbox(frame, self.bboxes[11-1], str(11))
            out_frame[2*h:3 * h, :w] = crop_frame_bbox(frame, self.bboxes[12 - 1], str(12))
            out_frame[:h, w:2*w] = crop_frame_bbox(frame, self.bboxes[9 - 1], str(9))
            out_frame[h:2 * h, w:2*w] = crop_frame_bbox(frame, self.bboxes[8 - 1], str(8))
            out_frame[2*h:3 * h, w:2*w] = crop_frame_bbox(frame, self.bboxes[7 - 1], str(7))
            out_frame[:h, 2*w:3 * w,] = crop_frame_bbox(frame, self.bboxes[2 - 1], str(2))
            out_frame[h:2 * h, 2*w:3 * w] = crop_frame_bbox(frame, self.bboxes[1 - 1], str(1))
            out_frame[2*h:3 * h, 2*w:3 * w] = crop_frame_bbox(frame, self.bboxes[6 - 1], str(6))
            out_frame[:h, 3 * w:4 * w] = crop_frame_bbox(frame, self.bboxes[3 - 1], str(3))
            out_frame[h:2 * h, 3 * w:4 * w] = crop_frame_bbox(frame, self.bboxes[4 - 1], str(4))
            out_frame[2*h:3 * h, 3 * w:4 * w] = crop_frame_bbox(frame, self.bboxes[5 - 1], str(5))
            out_frame[:h, 4 * w:5 * w,] = crop_frame_bbox(frame, self.bboxes[13 - 1], str(13))
            out_frame[h:2 * h, 4 * w:5 * w] = crop_frame_bbox(frame, self.bboxes[14 - 1], str(14))
            out_frame[2*h:3 * h, 4 * w:5 * w] = crop_frame_bbox(frame, self.bboxes[15 - 1], str(15))
            return out_frame
        else:
            print('bboxes sizes are not identical')
            return frame

    def export(self, filename):
        pd.DataFrame(self.data).to_csv(filename, index=False)
        print('Data exported to:', filename)

    def close(self):
        if self.export_pdf_filename:
            print('PDF report exported to:', self.export_pdf_filename)
            self.pdf_file.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    # Parameters
    folder = '/home/gilad/logs/reticle_target/3 mm iris/DOF 3MM IRIS'
    export_detected = True
    profile_flag = True
    resize = 0.25
    show = True
    export_pdf = False
    export_csv = True

    # Profiling
    profiler = cProfile.Profile()
    if profile_flag:
        profiler.enable()

    # Run CTF analysis
    files = os.listdir(folder)
    rgb32_files = [file for file in files if 'RGB32' in file and 'detected' not in file and 'report' not in file and 'data.csv' not in file]
    raw8_files = [file for file in files if 'RAW8' in file and 'detected' not in file and 'report' not in file and 'data.csv' not in file]

    for i, file in enumerate(rgb32_files):
        main_image = cv2.imread(os.path.join(folder, file))
        cv2.rotate(main_image, cv2.ROTATE_180, main_image)
        export_pdf_filename = os.path.join(folder, f'{file[:-4]}_report.pdf') if export_pdf else None
        export_csv_filename = os.path.join(folder, f'{file[:-4]}_data.csv') if export_csv else None
        ctf_analyser = CTFAnalyser(export_pdf_filename=export_pdf_filename)
        t0 = time.time()

        ctf_analyser.detect_targets(main_image, show=show, resize=resize)
        print('Analysis time:', int(1000 * (time.time() - t0)), '[ms]')
        if export_detected:
            cv2.imwrite(os.path.join(folder, f'{file[:-4]}_detected.bmp'), main_image)
            print("Detected target exported to:", os.path.join(folder, f'{file[:-4]}_detected.bmp'))
        ctf_analyser.export(export_csv_filename)
        ctf_analyser.close()
        break

    if profile_flag:
        profiler.disable()
        stats = pstats.Stats(profiler).sort_stats('cumtime')
        stats.strip_dirs()
        stats.print_stats()
        time_str = time.strftime("%Y-%m-%d_%H-%M-%S")
        stats.dump_stats('profile/profile_' + time_str + '.prof')
