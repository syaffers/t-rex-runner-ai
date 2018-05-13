import time
import cv2
import numpy as np
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
            image_gray, (image_gray.shape[1] // 2, image_gray.shape[0] // 2),
            interpolation=cv2.INTER_NEAREST)

        return image_resize


if __name__ == "__main__":
    print("Starting!")
    for i in range(5)[::-1]:
        print(i+1)
        time.sleep(1)

    x, y = 383, 173
    agent = KeyListeningAgent()

    action_lut = {
        "space": [1, 0, 0],
        "down": [0, 1, 0],
        "none": [0, 0, 1],
    }

    x_train = []
    y_train = []

    try:
        while True:
            im = grab_screen(x, y, x + 600, y + 150)
            im_processed = agent.process_image(im)

            x_train.append(im_processed)
            y_train.append(action_lut.get(agent.pressed_key))

            if len(x_train) % 1000 == 0:
                print(len(x_train))
                np.savez('data/training_half_2.npz',
                         images=np.array(x_train),
                         targets=np.array(y_train))


            # This needs to be here! Too fast and the OS will crash.
            time.sleep(1/100)

            if (cv2.waitKey(1) & 0xFF) == ord('q'):
                cv2.destroyAllWindows()
                break

    except KeyboardInterrupt:
        cv2.destroyAllWindows()
        agent.klg.cancel()
