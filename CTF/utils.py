import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
import cv2


def save_multi_image(filename, plt):
    pp = PdfPages(filename)
    fig_nums = plt.get_fignums()
    figs = [plt.figure(n) for n in fig_nums]
    for fig in figs:
        fig.savefig(pp, format='pdf')
    pp.close()


def non_max_suppression_fast(boxes, overlapThresh):
    # If there are no boxes, return an empty list
    if len(boxes) == 0:
        return []

    # Initialize the list of picked indexes
    pick = []

    # Grab the coordinates of the bounding boxes
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    # Compute the area of the bounding boxes and sort the bounding
    # boxes by the bottom-right y-coordinate of the bounding box
    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    idxs = np.argsort(y2)

    # Keep looping while some indexes still remain in the indexes list
    while len(idxs) > 0:
        # Grab the last index in the indexes list and add the index value to the list of
        # picked indexes
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)

        # Find the largest (x, y) coordinates for the start of the bounding box and the
        # smallest (x, y) coordinates for the end of the bounding box
        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])

        # Compute the width and height of the bounding box
        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)

        # Compute the ratio of overlap
        overlap = (w * h) / area[idxs[:last]]

        # Delete all indexes from the index list that have
        idxs = np.delete(idxs, np.concatenate(([last],
                                               np.where(overlap > overlapThresh)[0])))

    # Return only the bounding boxes that were picked
    return boxes[pick].astype("int")


def crop_frame_bbox(frame, bbox, text_at_center=None):
    bbox_center = ((bbox[3] - bbox[1]) // 2, (bbox[2] - bbox[0]) // 2)
    if (bbox[2] <= frame.shape[0] and
            bbox[0] <= frame.shape[0] and
            bbox[1] <= frame.shape[1] and
            bbox[3] <= frame.shape[1]):
        crop = frame[bbox[1]:bbox[3], bbox[0]:bbox[2], :].copy()
    else:
        crop = np.zeros((bbox[3] - bbox[1], bbox[2] - bbox[0], 3))
    if text_at_center is not None:
        crop = cv2.putText(crop, text_at_center, bbox_center, cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
    return crop
