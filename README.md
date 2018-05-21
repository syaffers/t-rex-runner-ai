# Adventures into AI: the Chrome T-Rex Runner

Tested on Ubuntu 16.04, with Python 3.5.4, OpenCV 3.1.0. Likely **not** to work
on Windows or Mac OSX.

## Quickstart

$$ \kappa i \because j \in \mathbb{X} $$

Compile the C library `prtscr.c` for fast screen capture in X11 environments
(courtesy of [this SO answer](https://stackoverflow.com/a/16141058])).

    gcc -shared -O3 -lX11 -fPIC -Wl,-soname,prtscn -o prtscn.so prtscn.c
    # use the command below if the above command doesn't work
    # gcc -shared -O3 -fPIC -Wl,-soname,prtscn -o prtscn.so prtscn.c -lX11

**Note:** The screen capture library can be too fast for some computers. Make
sure you add a delay using `time.sleep()` to prevent crashes if you are using
this library for your personal projects.

Run the agent, point to your browser window and run the game (you might want to
adjust the `FRAME_X` and `FRAME_Y` position of the screen capture window to
adapt for your monitor in `config.py`).

    python agent_cv.py

### Frame Sampling Rate Issues

It seems that the agents perform well on certain framerates. This is possibly
due to the sampling rate of the KL Agent during data collection which was\
configured with a `time.sleep()` delay of $\frac{1}{100}$. With the additional
preprocessing of the image, it is likely that additional lag was introduced
into the time delta between frames.

The optimal framerate that seems to be consistent over my PCs is
$\frac{1}{90}$. It may be better to try other values if the agent is not
performing well on yours.

## Dependencies

The T-rex runner game can be played on modern Google Chrome browsers when there
is no internet connectivity. But there are open-source versions thanks to the
[Chromium project](https://www.chromium.org/) and the distribution used for
this project is found at [wayou's GitHub project](https://github.com/wayou/t-rex-runner/).

The AI also uses typical Python libraries for matrix manipulations and keypress
interfaces:

- `numpy >= 1.12`: matrix handling
- `keras == 2.1.6`: deep learning library. *NB: Needed to load trained models.*
- `opencv >= 3.1.0`: fast image processing
- PyUserInput `== 0.1.11`: automating user input to global windows
- `pyxhook` for Python 3: global user inputs capturing for X11 environment

## How it Works

### Keyboard-Logging (KL) Agent

The KL Agent is not a game playing-agent, but instead is a data collection
agent which interfaces with the user inputs as they play the game. The agent
collects grayscale images of the game area (sized down to quarter size) and the
corresponding action at that frame. ~~The images and action vectors are then
stored in an `.npz` file.~~ Saving as `.npy` files are more performant.

### Computer Vision (CV) Agent

The CV Agent a simple computer vision AI which finds the bounding boxes of
approaching objects. If a cactus or a pterodactyl comes close enough to T-rex,
the agent will press `Space` to indicate a jumping action. It also accounts for
objects flying higher than the T-rex.

### Convolutional Net (CN) Agent

The CN agent uses labelled images to predict the action on a particular frame.
Using the training data collected by the KL agent, we have an image-action pair
which can be subjected to a normal supervised learning optimization. The agent
also shows the confidence of the predictions as bar plots on the top-left hand
corner of the agent's screen.

The convolutional net used in this agent is a variant of the ZF Net presented
by [Zeiler and Fergus (2013)](https://arxiv.org/abs/1311.2901) with the number
of filters reduced to fit the problem at hand. This conv net predicts on a
single image frame the action that needs to be executed. More details of the
net can be found by exploring the model in Keras.

Another variant of the CN agent is also implemented to predict the action of
the the lastest of 5 frames. The 5 most recent frames are stacked and then
passed through the net. This convolutional net was inspired by
[Mnih et al. (2013)](https://arxiv.org/abs/1312.5602); who featured a smaller
convolutional network (than ZF Net) and is used in a Q-learning algorithm.
Turned out it works well for supervised learning too. *This is the best
performing agent so far.*

To switch between the two agents, switch out the commented lines in the bottom
section of `agent_cn.py`.

## Trained Agent Models

Trained models can be found in the recent [releases](https://github.com/syaffers/t-rex-runner-ai/releases).

## Wishlist

This started as a OpenCV weekend exercise which went real. I want to see what
the limits of ML are just based on captured images, rather than having
"physics data" from the actual game. Things I'd like to try out:

- [x] Train an OpenCV agent.
  - *Implemented `agent_cv.py`.*

- [x] Train a conv net with supervised learning using ~50000 images.
  - *Implemented `agent_cn.py`, and trained the `DinoBot.h5` and `DinoBotS.h5`
  Keras models.*
  - *Made smaller conv net models of `DinoBot` and `DinoBotS`.*

- [ ] Train a conv net with policy gradients.
