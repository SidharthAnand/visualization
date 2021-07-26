# Butlr Visualization and Labeling

## Getting Started

### Running command arguments:

-path: Path to saved data file (in .txt format)

-det: Path to saved detection/bounding box data (in .txt format)

-mqdi: MQTT data channel (for live-streaming data)

-mqba: MQTT broker address

-fps: FPS; 8 by default

-sz: The size in pixels of the data window (600 by default. This has worked well, so I would recommend leaving it alone)

-lbl: Activates labeling functionality (t/f)

### Sample running command 
To visualize the data and edit the bounding boxes included in the **visualizer/test_data** folder of the repo, simply navingate to the **visualizer** foler and run

_python3 visualizer.py -path test_data/lying_29_32x32_sensor.txt -det test_data/lying_29_32x32_sensor_boundingbox.txt -lbl t_

## Visualization

### Live Data
Specify MQTT broker address and MQTT channel with *-mqba* and *mqdi*, respectively, to inspect live-streaming data.
<img width="963" alt="Screen Shot 2021-07-14 at 10 35 36 AM" src="https://user-images.githubusercontent.com/68759595/125667056-26858361-6b4e-4ff9-8d38-68bcf77547a7.png">


### Saved Data

<img width="958" alt="Screen Shot 2021-07-14 at 10 04 07 AM" src="https://user-images.githubusercontent.com/68759595/125663259-295bceba-ce0b-455e-8038-fe87d7e0e4fe.png">

#### Features
- Adjust Contrast: Change the values of pixels that are mapped to the darkest cold value (blue/black) and to the brightest hot value (yellow/white). Defaults to a range of 65 - 115 (degrees C * 4)
- Toggle color: Switch between visualization in grayscale and in color (Keyboard shortcut: c)
- Playback controls: Scrub through the file with the horizontal scroll bar at the bottom of the screen, play/pause the visualization, skip forward/backward 100 frames (Keyboard shortcuts: play/pause - spacebar, FF - right arrow, RW - left arrow)
- Time: The timestamp of the current frame in the visualization is displayed in the top right corner of the window

### Labeling
To edit the labels of an existing bounding box label file, be sure to add the path to the running command using the *-det* argument. Also don't forget to set *-lbl t*!

The visualization will behave as normal until the *Edit Bounding Box* button is pressed.
<img width="954" alt="Screen Shot 2021-07-14 at 10 15 12 AM" src="https://user-images.githubusercontent.com/68759595/125664499-21ec97c0-8e56-463f-a6c6-b836fdf22fc6.png">

Once the Editing process is initiated, three new buttons will appear:

- Previous Frame
- Next Frame
- Clear

And *Edit Bounding Box* will become *Done Editing*

<img width="964" alt="Screen Shot 2021-07-14 at 10 14 52 AM" src="https://user-images.githubusercontent.com/68759595/125664986-49662e44-c48c-4c35-8fb1-16b186db6be7.png">

While you are editing the bounding box data, you will be able to scrub forward and backward with the scroll bar as well as adjust the current frame with FF, RW, Previous Frame, and Next frame, but play functionality will be disabled until *Done Editing* has been pressed.

To add a bounding box, just click on the data window to specify the TOP LEFT and BOTTOM RIGHT corners of the box.

Pressing *Done Editing* does **NOT** save the file and quit out of the visualizer; it just stops the editing process until it is pressed again.

Once you are finished editing the bounding box data, quit out of the visualizer to save the file using *Exit* or the escape key. A new file will be written to the same path as your original detection text file, but with the suffix *_EDITED* attached to denote that it has been checked by hand. The new file will maintain all bounding boxes that were not edited, even if you did not scroll through them in the visualizer. If you wish to rewrite the file, **DELETE THE EXISTING FILE BEFORE CREATING A NEW ONE WITH THE VISUALIZER'S LABELING TOOL**, or just pass in the *_EDITED* file to the tool in the running command to restart the process with the file you just finished editing.
