import time
from collections import deque
import cv2
import keras
import numpy as np
from pykeyboard import PyKeyboard
from agent_cn import ConvNetAgent
from config import FRAME_X, FRAME_Y
from grabber import grab_screen


class ConvNetStridedAgent(ConvNetAgent):
    def __init__(self):
        self.action_lut = ['Jump', 'Duck', 'Run']
        self.action_colors = [
            (0, 128, 0),
            (0, 0, 255),
            (255, 0, 0)
        ]
        self.memory = deque([], maxlen=5)
        self.conv_net = keras.models.load_model('DinoBotS_v1.h5')
        self.keyboard = PyKeyboard()
        self.pressed_key = "."

    def handle_key(self, action):
        """ Key press handler for each action. """
        if action == 0:
            self.pressed_key = " "
            self.keyboard.press_key(" ")
        elif action == 1:
            self.pressed_key = self.keyboard.down_key
            self.keyboard.press_key(self.keyboard.down_key)
        else:
            self.keyboard.release_key(self.pressed_key)

    def act(self, threshold=0.5):
        """ Decision-making happens here. Based on the input image, the conv
        net needs to decide on the best course of action.
        """
        # Reshape for Keras.
        work_image = np.rollaxis(np.array(self.memory), 0, 3).reshape((-1, 37, 150, 5))
        action_confidences = self.conv_net.predict(work_image).flatten()

        # If very confident, then take the action. Otherwise, just run.
        # TODO: Could be better if we could proceed with last confident action.
        if np.any(action_confidences > threshold):
            self.handle_key(action_confidences.argmax())
        else:
            self.handle_key(2)

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
            interpolation=cv2.INTER_NEAREST)

        self.memory.append(image_resize)

        if len(self.memory) >= 5:
            # Predict the action.
            action_confidences = self.act()
            # Print confidence barplots on image.
            for i, confidence in enumerate(action_confidences):
                cv2.rectangle(
                    work_image, (10*i+10, int(60-50*confidence)), (10*i+19, 60),
                    self.action_colors[i], -1
                )
                cv2.putText(work_image, self.action_lut[i][0], (10*i+10, 72),
                            cv2.FONT_HERSHEY_PLAIN, 1, self.action_colors[i])

        return work_image


if __name__ == "__main__":
    agent = ConvNetStridedAgent()
    main_window_name = "ConvNetStridedAgent"
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

            if cv2.waitKey(1) & 0xFF == ord('q'):
                cv2.destroyAllWindows()
                agent.handle_key(2)
                break

    except KeyboardInterrupt:
        cv2.destroyAllWindows()
