# Template matching is a technique used to find a template image in a larger image.
import cv2
import numpy as np

def non_max_suppression_fast(boxes, overlapThresh):
    # If there are no boxes, return an empty list
    if len(boxes) == 0:
        return []

    # Initialize the list of picked indexes
    pick = []

    # Grab the coordinates of the bounding boxes
    x1 = boxes[:,0]
    y1 = boxes[:,1]
    x2 = boxes[:,2]
    y2 = boxes[:,3]

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


cv2.namedWindow('img', cv2.WINDOW_NORMAL)
cv2.resizeWindow('img', 1280, 720)
# Load the main image and template
main_image = cv2.imread('/home/gilad/logs/reticle_target/12 mm iris/DOF IRIS 12MM DOWN/Left Camera IRIS 12MM RGB32 exp 2ms H 0(2)(94.5).bmp')
template = cv2.imread('/home/gilad/logs/reticle_target/12 mm iris/DOF IRIS 12MM DOWN/template.bmp')
h, w = template.shape[:2]

# Convert images to grayscale
main_image_gray = cv2.cvtColor(main_image, cv2.COLOR_BGR2GRAY)
template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)

# Perform template matching
res = cv2.matchTemplate(main_image_gray, template_gray, cv2.TM_CCOEFF_NORMED)
threshold = 0.8
loc = np.where(res >= threshold)

# Create a list of bounding boxes
boxes = []
for pt in zip(*loc[::-1]):
    boxes.append([pt[0], pt[1], pt[0] + w, pt[1] + h])

boxes = np.array(boxes)

# Apply non-maximum suppression
nms_boxes = non_max_suppression_fast(boxes, 0.3)

# Draw rectangles around the matches
for (startX, startY, endX, endY) in nms_boxes:
    cv2.rectangle(main_image, (startX, startY), (endX, endY), (0, 255, 0), 2)

# Display the result
cv2.imshow('img', main_image)
cv2.waitKey(0)
cv2.destroyAllWindows()