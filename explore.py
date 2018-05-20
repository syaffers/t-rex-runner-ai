import os
import time
import cv2
import numpy as np
# from grabber import grab_screen
# from config import FRAME_X, FRAME_Y

####################
# GRAB A SCREENSHOT
####################
# y = 173
# time.sleep(5)
# im = grab_screen(FRAME_X, FRAME_Y, FRAME_X + 600, FRAME_Y + 150)
# cv2.imwrite('gameover4.png', np.array(im))

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

fps = 30
last_n = -1000
for i, t in zip(images[last_n:], targets[last_n:]):
    img = cv2.putText(i, str(fps), (10, 10), font, 1, (0xFF, 0xFF, 0xFF), 1)
    img = cv2.putText(i, str(t), (60, 10), font, 1, (0xFF, 0xFF, 0xFF), 1)
    cv2.imshow('window', cv2.resize(i, (600, 150), interpolation=cv2.INTER_NEAREST))

    kp = cv2.waitKey(1) & 0xFF

    if kp == ord('q'):
        cv2.destroyAllWindows()
        break

    if kp == ord('w'):
        fps += 5.

    if kp == ord('s'):
        fps -= 5.

    time.sleep(1/fps)

cv2.destroyAllWindows()
