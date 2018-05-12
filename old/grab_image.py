import numpy as np
from pyscreenshot import grab
import cv2
import time

last_time = time.time()

while(True):
    try:
        x, y = 383, 173
        ps_pil = grab(bbox=(x, y, x+600, y+150))

        print("Took {}s".format(time.time() - last_time))
        last_time = time.time()
        ps_gray = cv2.cvtColor(np.array(ps_pil), cv2.COLOR_BGR2GRAY)
        cv2.imshow('window', ps_gray)

        if cv2.waitKey(25) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break

    except Exception:
        cv2.destroyAllWindows()
        break
