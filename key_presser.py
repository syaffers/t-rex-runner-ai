import time
import random
from pykeyboard import PyKeyboard

k = PyKeyboard()

for i in range(5)[::-1]:
    print(i)
    time.sleep(1)

while True:
    try:
        k.press_key(' ')
        time.sleep(0.01)
        k.release_key(' ')
        time.sleep(random.random() * 3)
    except KeyboardInterrupt:
        print("Ending...")
        break

