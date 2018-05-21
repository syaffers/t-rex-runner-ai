from collections import deque
import time
import cv2
import keras
import numpy as np
from pykeyboard import PyKeyboard
from config import FRAME_X, FRAME_Y
from grabber import grab_screen


class ConvNetAgent(object):
    def __init__(self, keras_model, memory_size):
        """ Agent constructor. Takes arguments of 1) path to the desired model,
        and 2) the memory size of the convolutional net.
        """
        self.action_lut = ['J', 'D', 'R']
        self.action_colors = [
            (0, 128, 0),  # Dark green
            (0, 0, 255),  # Red
            (255, 0, 0)   # Blue
        ]
        self.conv_net = keras_model
        self.is_paused = False
        self.keyboard = PyKeyboard()
        self.memory_size = memory_size
        self.memory = deque([], maxlen=memory_size)
        self.pressed_key = "."

    def handle_keypress(self, key):
        """ Keypress handler. We probably don't want to erratically press keys.
        If the registered key on the current frame is the key that is currently
        pressed, don't release key.
        """
        if self.pressed_key is not key:
            self.keyboard.release_key(self.pressed_key)
            self.pressed_key = key

        self.keyboard.press_key(self.pressed_key)

    def handle_action(self, action):
        """ Key press handler for each action. """
        if action == 0:
            self.handle_keypress(" ")
        elif action == 1:
            self.handle_keypress(self.keyboard.down_key)
        else:
            self.keyboard.release_key(self.pressed_key)

    def act(self, threshold=0.5):
        """ Decision-making happens here. Based on the input image, the conv
        net needs to decide on the action it is most confident with. Threshold
        the confidence so it's not too erratic.
        """
        # Reshape for Keras.
        work_image = np.rollaxis(np.array(self.memory), 0, 3)
        work_image = work_image.reshape((1, 37, 150, self.memory_size))
        action_confidences = self.conv_net.predict(work_image).flatten()

        # If very confident, then take the action. Otherwise, just run.
        if np.any(action_confidences > threshold):
            self.handle_action(action_confidences.argmax())
        else:
            self.handle_action(2)

        return action_confidences

    def process_image(self, image):
        """ Agent's image processing routine. """
        # Working image.
        work_image = np.array(image)
        # Grayed-out.
        image_gray = cv2.cvtColor(work_image, cv2.COLOR_RGB2GRAY)
        # Inverting to make objects easier to "see".
        image_gray = cv2.bitwise_not(image_gray)
        # Resize for neural net.
        image_resize = cv2.resize(
            image_gray, (image_gray.shape[1] // 4, image_gray.shape[0] // 4),
            interpolation=cv2.INTER_NEAREST
        )

        self.memory.append(image_resize)

        # Pause for safety.
        if self.is_paused:
            cv2.putText(work_image, "Paused", (10, 10),
                        cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255))
            return work_image

        if len(self.memory) >= self.memory_size:
            # Predict the action.
            action_confidences = self.act()
            # Print confidence barplots on image.
            for i, confidence in enumerate(action_confidences):
                bar_x1 = 10 * i + 10
                bar_x2 = 10 * i + 19
                bar_y1 = int(60 - 50 * confidence)
                bar_y2 = 60

                cv2.rectangle(work_image, (bar_x1, bar_y1), (bar_x2, bar_y2),
                            self.action_colors[i], -1)
                cv2.putText(work_image, self.action_lut[i][0], (10*i+10, 72),
                            cv2.FONT_HERSHEY_PLAIN, 1, self.action_colors[i])

        return work_image


if __name__ == "__main__":
    model = keras.models.load_model('./models/DinoBotS_v2.h5')
    channel_size = model.input.shape[-1].value
    agent = ConvNetAgent(model, channel_size)

    main_window_name = "ConvNetAgent"
    cv2.namedWindow(main_window_name)
    cv2.moveWindow(main_window_name, 150, 400)

    print("Starting!")
    for i in range(5)[::-1]:
        print(i+1)
        time.sleep(1)

    try:
        while True:
            im = grab_screen(FRAME_X, FRAME_Y, FRAME_X + 600, FRAME_Y + 150)
            im_processed = agent.process_image(im)
            cv2.imshow(main_window_name, im_processed)

            # This needs to be here! Too fast and the OS will crash.
            time.sleep(1/90.)

            cv_key = cv2.waitKey(1) & 0xFF

            if cv_key == ord('s'):
                agent.is_paused = not agent.is_paused
                agent.keyboard.release_key(agent.pressed_key)

            if cv_key == ord('q'):
                cv2.destroyAllWindows()
                agent.keyboard.release_key(agent.pressed_key)
                break

    except KeyboardInterrupt:
        cv2.destroyAllWindows()
        agent.keyboard.release_key(agent.pressed_key)
