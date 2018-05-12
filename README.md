# Adventures into AI: the Chrome Dinosaur Jumper

Tested on Ubuntu 16.04, with Python 3.5.4, OpenCV 3.1.0.

## Quickstart

Compile C library to capture screen faster than other Python capture libraries
(courtesy of [this SO answer](https://stackoverflow.com/a/16141058]))

    gcc -shared -O3 -lX11 -fPIC -Wl,-soname,prtscn -o prtscn.so prtscn.c
    # use the command below if the above command doesn't work
    # gcc -shared -O3 -fPIC -Wl,-soname,prtscn -o prtscn.so prtscn.c -lX11

**Note:** The screen capture library can be too fast for some computers. Make
sure you add a delay using `time.sleep()` to prevent crashes.

Run the agent and point to your browser window (you might want to adjust the
`x` and `y` position for your monitor in `agent.py`).

    python agent.py
