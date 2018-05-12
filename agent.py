import time
import numpy as np
import cv2
from pykeyboard import PyKeyboard
from grabber import grab_screen
from image_tricks import auto_canny_params


def process_image(image, keyboard, offset=66):
    work_image = np.array(image)

    image_gray = cv2.cvtColor(work_image, cv2.COLOR_RGB2GRAY)
    image_gray[:, :offset] = 255

    image_blur = cv2.GaussianBlur(image_gray, (7, 7), 0)

    lower, upper = auto_canny_params(image_blur)
    image_edge = cv2.Canny(image_blur, lower, upper, apertureSize=3)

    _, contours, _ = cv2.findContours(
        image_edge, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

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

    register_jump = False

    if (closest_x - offset) < 70:
        register_jump = True

    if closest_y < 92:
        register_jump = False

    print(closest_y)

    if register_jump:
        keyboard.press_key(' ')
        time.sleep(0.15)
        keyboard.release_key(' ')

    return work_image


if __name__ == "__main__":
    k = PyKeyboard()

    print("Starting!")
    for i in range(5)[::-1]:
        print(i+1)
        time.sleep(1)

    main_window_name = "Agent"
    cv2.namedWindow(main_window_name)
    cv2.moveWindow(main_window_name, 150, 400)

    while True:
        x, y = 383, 173
        im = grab_screen(x, y, x + 600, y + 150)

        im_processed = process_image(im, k)
        cv2.imshow(main_window_name, im_processed)

        # This needs to be here! Too fast and the OS will crash.
        time.sleep(1/50)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break
