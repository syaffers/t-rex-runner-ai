import time
import numpy as np
import cv2
from pykeyboard import PyKeyboard
from config import FRAME_X, FRAME_Y
from grabber import grab_screen
from image_tricks import auto_canny_params


class CVAgent(object):
    def __init__(self):
        self.is_key_pressed = False
        self.register_jump = False
        self.jump_frames = 0
        self.keyboard = PyKeyboard()
        self.last_closest_x = 0
        self.last_closest_y = 0

    def long_press_key(self, key):
        """ Kind-of async long-press jump function. Agent will press space for
        a few frames before letting go, simulating a long-press.
        """
        self.jump_frames -= 1
        if self.jump_frames <= 0 and self.is_key_pressed:
            self.is_key_pressed = False
            self.register_jump = False
            self.keyboard.release_key(key)
            return

        if not self.is_key_pressed:
            self.is_key_pressed = True
            self.keyboard.press_key(key)
            return

        return

    def jump(self, closest_x, closest_y, offset):
        """ Jump function to separate image processing and agent logic. Agent
        shouldn't be able to register any actions if an action is currently
        being run (hence why we have those `not self.is_key_pressed`)
        """
        if (closest_x - offset) < 72 and not self.is_key_pressed:
            self.register_jump = True
            self.jump_frames = 5

        if closest_y < 92 and not self.is_key_pressed:
            self.register_jump = False

        if self.register_jump:
            self.long_press_key(' ')

    def process_image(self, image, offset=66):
        """ Agent image processing routine """
        # Working image.
        work_image = np.array(image)
        # Grayed-out.
        image_gray = cv2.cvtColor(work_image, cv2.COLOR_RGB2GRAY)

        font = cv2.FONT_HERSHEY_PLAIN

        # Game over template.
        go_template = cv2.imread('assets/go_template.png')
        go_template = cv2.cvtColor(go_template, cv2.COLOR_BGR2GRAY)

        # Template matching.
        tm_threshold = 1500000
        match = cv2.matchTemplate(image_gray, go_template, cv2.TM_SQDIFF)
        score, _, _, _ = cv2.minMaxLoc(match)

        # Check if game is over before proceeding.
        if score < tm_threshold:
            cv2.putText(work_image, 'Game Over', (20, 20), font, 1, (0, 0, 255))
            return work_image

        # Crop off the dino part of the image.
        image_gray[:, :offset] = 255

        # Detect edges of objects.
        image_blur = cv2.GaussianBlur(image_gray, (7, 7), 0)
        lower, upper = auto_canny_params(image_blur)
        image_edge = cv2.Canny(image_blur, lower, upper, apertureSize=3)

        # Find contours.
        _, contours, _ = cv2.findContours(
            image_edge, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Determine the closest object to dino.
        closest_x = 600
        closest_y = 0
        for contour in contours:
            box_x, box_y, box_w, box_h = cv2.boundingRect(contour)

            if box_w + box_h > 10:
                cv2.rectangle(work_image,
                            (box_x, box_y),
                            (box_x + box_w, box_y + box_h),
                            (0, 128, 0), 2)

                if box_x < closest_x:
                    closest_x = box_x

                if (box_y + box_h) > closest_y:
                    closest_y = box_y + box_h

        self.last_closest_x = closest_x
        self.last_closest_y = closest_y

        # Let's jump!
        self.jump(closest_x, closest_y, offset)

        return work_image


if __name__ == "__main__":
    agent = CVAgent()
    main_window_name = "CompVisionAgent"
    cv2.namedWindow(main_window_name)
    cv2.moveWindow(main_window_name, 150, 400)

    print("Starting!")
    for i in range(5)[::-1]:
        print(i+1)
        time.sleep(1)

    while True:
        im = grab_screen(FRAME_X, FRAME_Y, FRAME_X + 600, FRAME_Y + 150)

        im_processed = agent.process_image(im)
        cv2.imshow(main_window_name, im_processed)

        # This needs to be here! Too fast and the OS will crash.
        time.sleep(1/50)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break
