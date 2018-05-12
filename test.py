import time
import cv2
import numpy as np
from grabber import grab_screen
from image_tricks import auto_canny_params

x, y = 383, 173
# time.sleep(5)
# im = grab_screen(x, y, x + 600, y + 150)
im = cv2.imread('test.png')

image = np.array(im)
image_gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
image_gray[:, :66] = 255

image_blur = cv2.GaussianBlur(image_gray, (7, 7), 0)

low, hi = auto_canny_params(image_blur)
image_edge = cv2.Canny(image_blur, low, hi, apertureSize=3)

_, contours, hierarchy = cv2.findContours(
    image_edge, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

for contour in contours:
    x, y, w, h = cv2.boundingRect(contour)

    if w + h > 10:
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 128, 0), 2)

# cv2.imshow('window', image)

# while True:
#     if cv2.waitKey(25) & 0xFF == ord('q'):
#         cv2.destroyAllWindows()
#         break
