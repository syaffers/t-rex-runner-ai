# Source: https://stackoverflow.com/questions/69645/take-a-screenshot-via-a-python-script-linux
# by JHolta

import ctypes
import os
import time
import cv2
import numpy as np
from PIL import Image

lib_name = 'prtscn.so'
abs_path_lib = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), lib_name)
grab = ctypes.CDLL(abs_path_lib)

def grab_screen(x1, y1, x2, y2):
    w, h = x2 - x1, y2 - y1
    size = w * h
    obj_length = size * 3

    grab.getScreen.argtypes = []
    result = (ctypes.c_ubyte * obj_length)()

    grab.getScreen(x1, y1, w, h, result)

    return Image.frombuffer('RGB', (w, h), result, 'raw', 'RGB', 0, 1)

if __name__ == '__main__':
    last_time = time.time()

    while True:
        x, y = 383, 173
        im = grab_screen(x, y, x+600, y+150)

        print("Took {}s".format(time.time() - last_time))
        last_time = time.time()
        im_gray = cv2.cvtColor(np.array(im), cv2.COLOR_RGB2GRAY)
        cv2.imshow('window', im_gray)

        time.sleep(1/30)

        if cv2.waitKey(25) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break
