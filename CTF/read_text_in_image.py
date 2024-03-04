import cv2
import pytesseract
import cv2
import pytesseract

cv2.namedWindow('img', cv2.WINDOW_NORMAL)
cv2.resizeWindow('img', 1280, 720)
img = cv2.imread('/home/gilad/logs/reticle_target/12 mm iris/DOF IRIS 12MM DOWN/Left Camera IRIS 12MM RGB32 exp 2ms H 0(2)(94.5).bmp')

h, w, c = img.shape
boxes = pytesseract.image_to_boxes(img)
for b in boxes.splitlines():
    b = b.split(' ')
    img = cv2.rectangle(img, (int(b[1]), h - int(b[2])), (int(b[3]), h - int(b[4])), (0, 255, 0), 2)

cv2.imshow('img', img)
cv2.waitKey(0)



# Load the image
image_path = '/home/gilad/logs/reticle_target/12 mm iris/DOF IRIS 12MM DOWN/Left Camera IRIS 12MM RGB32 exp 2ms H 0(2)(94.5).bmp'
image = cv2.imread(image_path)

# Convert the image to grayscale
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Apply thresholding to make the image binary
_, binary_image = cv2.threshold(gray_image, 127, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

# Optionally apply additional preprocessing (e.g., dilation, erosion) to improve OCR accuracy

# Configure pytesseract path (if necessary)
# pytesseract.pytesseract.tesseract_cmd = r'<full_path_to_your_tesseract_executable>'

cv2.imshow('img', binary_image)
cv2.waitKey(0)

# Perform OCR
text = pytesseract.image_to_string(gray_image, config=r'--oem 3 --psm 6 outputbase digits')

print(text)