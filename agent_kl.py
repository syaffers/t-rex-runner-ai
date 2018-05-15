import time
import datetime
import cv2
import numpy as np
from config import FRAME_X, FRAME_Y
from grabber import grab_screen
from pyxhook import HookManager


class KeyListeningAgent(object):
    def __init__(self):
        self.klg = HookManager()
        self.klg.KeyDown = self.key_down
        self.klg.KeyUp = self.key_up
        self.klg.HookKeyboard()
        self.klg.start()
        self.pressed_key = "none"

    def key_down(self, event):
        k = event.Key
        self.pressed_key = k.lower()

    def key_up(self, event):
        self.pressed_key = "none"

    def process_image(self, image):
        # Working image.
        work_image = np.array(image)
        # Grayed-out.
        image_gray = cv2.cvtColor(work_image, cv2.COLOR_RGB2GRAY)
        # Inverting to make objects easier to "see".
        image_gray = cv2.bitwise_not(image_gray)
        # Resize for neural net.
        image_resize = cv2.resize(
            image_gray, (image_gray.shape[1] // 4, image_gray.shape[0] // 4),
            interpolation=cv2.INTER_NEAREST)

        return image_resize


if __name__ == "__main__":
    agent = KeyListeningAgent()

    action_lut = {
        "space": [1, 0, 0],
        "up": [1, 0, 0],
        "down": [0, 1, 0],
        "none": [0, 0, 1],
    }

    timestamp = datetime.datetime.now()
    prefix = timestamp.strftime("%Y%m%d%H%M%S")

    x_train = []
    y_train = []
    batch = 0

    print("Starting!")
    for i in range(5)[::-1]:
        print(i+1)
        time.sleep(1)

    try:
        while True:
            im = grab_screen(FRAME_X, FRAME_Y, FRAME_X + 600, FRAME_Y + 150)
            im_processed = agent.process_image(im)

            x_train.append(im_processed)
            y_train.append(action_lut.get(agent.pressed_key))

            if len(x_train) % 500 == 0:
                print(len(x_train))
                batch += 1
                save_path = 'data/{}_training_qrtr_{}_{}.npy'
                np.save(save_path.format(prefix, "%03d" % batch, "x"), x_train)
                np.save(save_path.format(prefix, "%03d" % batch, "y"), y_train)

                x_train = []
                y_train = []

            # This needs to be here! Too fast and the OS will crash.
            time.sleep(1/100.)

    except KeyboardInterrupt:
        cv2.destroyAllWindows()
        agent.klg.cancel()
