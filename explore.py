import os
import time
import cv2
import numpy as np
# from grabber import grab_screen
# from image_tricks import auto_canny_params

####################
# GRAB A SCREENSHOT
####################
# x, y = 660, 173
# time.sleep(5)
# im = grab_screen(x, y, x + 600, y + 150)
# cv2.imwrite('gameover4.png', np.array(im))

##############
# LOGIC STUFF
##############
# im = cv2.imread('assets/test.png')

# image = np.array(im)
# image_gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
# image_gray[:, :66] = 255

# go_template = cv2.imread('assets/go_template.png')
# go_template = cv2.cvtColor(go_template, cv2.COLOR_BGR2GRAY)

# res = cv2.matchTemplate(image_gray, go_template, cv2.TM_SQDIFF)
# print(cv2.minMaxLoc(res))

# image_blur = cv2.GaussianBlur(image_gray, (7, 7), 0)


# low, hi = auto_canny_params(image_sharp)
# image_edge = cv2.Canny(image_sharp, low, hi, apertureSize=3)

# _, contours, hierarchy = cv2.findContours(
#     image_edge, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# for contour in contours:
#     x, y, w, h = cv2.boundingRect(contour)

#     if w + h > 10:
#         cv2.rectangle(image, (x, y), (x + w, y + h), (0, 128, 0), 2)

# cv2.imshow('window', image_gray)

# while True:
#     if cv2.waitKey(25) & 0xFF == ord('q'):
#         cv2.destroyAllWindows()
#         break

#########################
# EXPLORE COLLECTED DATA
#########################
data_path = './data/'
images, targets = None, None
for filename in sorted(os.listdir(data_path)):
    data = np.load(os.path.join(data_path, filename))

    if '_x' in filename:
        if images is None:
            images = data
        else:
            images = np.vstack([images, data])
    else:
        if targets is None:
            targets = data
        else:
            targets = np.vstack([targets, data])

del data

font = cv2.FONT_HERSHEY_PLAIN

for i, t in zip(images[-5000:], targets[-5000:]):
    img = cv2.putText(i, str(t), (20, 20), font, 1, (0xFF, 0xFF, 0xFF), 2)
    cv2.imshow('window', i)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    time.sleep(1/60.)

cv2.destroyAllWindows()
