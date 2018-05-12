# Adventures into AI: the Chrome T-Rex Runner

Tested on Ubuntu 16.04, with Python 3.5.4, OpenCV 3.1.0.

## Quickstart

Compile the C library `prtscr.c` for fast screen capture in X11 environments
(courtesy of [this SO answer](https://stackoverflow.com/a/16141058])).

    gcc -shared -O3 -lX11 -fPIC -Wl,-soname,prtscn -o prtscn.so prtscn.c
    # use the command below if the above command doesn't work
    # gcc -shared -O3 -fPIC -Wl,-soname,prtscn -o prtscn.so prtscn.c -lX11

**Note:** The screen capture library can be too fast for some computers. Make
sure you add a delay using `time.sleep()` to prevent crashes. `agent.py` has
this already coded-in.

Run the agent, point to your browser window and run the game (you might want to
adjust the `x` and `y` position of the screen capture window to adapt for your
monitor in `agent.py`).

    python agent.py

## Dependencies

The T-rex runner game can be played on modern Google Chrome browsers when there
is no internet connectivity. But there are open-source versions thanks to the
[Chromium project](https://www.chromium.org/) and the distribution used for
this project is found at [wayou's GitHub project](https://github.com/wayou/t-rex-runner/).

## How it Works

It's a simple computer vision AI which finds the bounding boxes of approaching
objects. If a cactus or a pterodactyl comes close enough to T-rex, the code
will press `Space` to indicate a jumping action. It also accounts for objects
flying higher than the T-rex.

## Wishlist

This started as a OpenCV weekend exercise which went real. I want to see what
the limits of ML are just based on captured images, rather than having
"physics data" from the actual game. Things I'd like to try out:

- Train a convolutional neural network to predict the actions. Possibly with
  policy gradients.
