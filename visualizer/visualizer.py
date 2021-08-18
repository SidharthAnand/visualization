import json
import re
import os
from datetime import timedelta

import cv2
import ast
import sys
import time
import pygame
import _thread
import argparse
import colorsys
import pygame_gui
import numpy as np
from collections import deque
from pygame import mixer
from datetime import datetime
import paho.mqtt.client as paho

max_hue = 240 * 0.00277777777777


def closest(list, Number):
    aux = []
    for valor in list:
        aux.append(abs(Number - valor))

    return aux.index(min(aux))


def visualizer(
    data_path=None,
    detection_path=None,
    live=False,
    aspect_ratio=(600, 600),
    fps=8,
    data_topic=None,
    mqtt_address=None,
    label=False,
    mac="xxx",
    unified=False,
    restream=None,
    time_start=None,
    time_end=None,
    time_offset=None,
    db=None,
    hid=None,
    sensor=None,
    normalized=None,
    snap=None,
    gtVidPath=None,
):
    global playback
    global text_line
    global bb_lbl_dict
    global data_queue
    global detection_queue
    global det_index
    global det_curr
    global text_length
    global client
    global pause
    global timestamp
    global selected_pos
    global det_out_file
    global times
    global pos_drop
    global category_ids
    global category_options
    global vizMac

    timestamp = time.time()
    vizMac = mac
    client = paho.Client()
    pause = label
    norm = normalized

    pygame.init()

    labeling_active = False
    texts = []
    bb_lbl = []
    bb_lbl_dict = {}
    points = []
    new_boxes = []
    sensor_mac = ""

    white = (255, 255, 255)
    black = (0, 0, 0)
    red = (255, 0, 0)
    clock = pygame.time.Clock()
    running = True
    smaller_size = True
    if smaller_size:
        grid = (1.5, 1, aspect_ratio[0])
    else:
        grid = (2, 1, aspect_ratio[0])
    scheme = "thermal"

    box = grid[2]
    width = grid[0] * box
    height = box * grid[1]
    aspect = (
        int(width + (aspect_ratio[0] // 10)),
        int(height + (aspect_ratio[1] // 10)),
    )
    font_size_small = 12
    font_size_large = int(grid[2] // 30)
    det_index = 0
    det_curr = 0

    # error checking for time restream
    try:
        if restream:
            time_offset = float(time_offset)
    except:
        print("ERROR: Time offset not valid")
        quit()
    if not date_format(time_start) and restream:
        print("ERROR: Start time in wrong format")
        quit()
    elif not date_format(time_end) and time_end is not None and restream:
        print("ERROR: End time in wrong format")
        quit()

    is_restream = restream and (isinstance(time_offset, float))

    category_options = [
        "Uncertain",
        "Sitting",
        "Standing",
        "Lying Down",
    ]  # change the title for whether the person is unsure
    category_ids = [-1, 0, 1, 2]
    selected_pos = -1
    category_options = [
        "Uncertain",
        "Sitting",
        "Standing",
        "Lying Down",
    ]  # change the title for whether the person is unsure
    category_ids = [-1, 0, 1, 2]

    if not smaller_size:
        br = 0.88
        lb = 0.84
        hb = 0.839

    else:
        br = 0.8
        lb = 0.75
        hb = 0.745
        plb = 0.83
    try:
        font = pygame.font.Font("../font/Consola.ttf", font_size_small)
        title_font = pygame.font.Font("../font/Consola.ttf", font_size_large)
    except FileNotFoundError:
        font = pygame.font.Font("font/Consola.ttf", font_size_small)
        title_font = pygame.font.Font("font/Consola.ttf", font_size_large)
    if not live:
        cap = data_path
    else:
        cap = data_topic
    pygame.display.set_caption(f"Butlr Data Visualization -- {cap}")
    if scheme == "thermal":
        color = lambda x, y, z: thermal(x, y, z)
        color_on = True
    elif scheme == "grayscale":
        color = lambda x, y, z: grayscale(x, y, z)
        color_on = False
    else:
        raise ValueError

    low_bound = 65
    high_bound = 115

    data_queue = deque([])
    detection_queue = deque([])

    multiplier = 1
    dispX = aspect[0]
    if gtVidPath:
        c = cv2.VideoCapture(gtVidPath)
        total_frames = c.get(7)
        success, img = c.read()
        shape = img.shape[1::-1]
        # print(img.shape)
        assert box < shape[0] and box < shape[1]
        cx = 120
        cy = 400
        # print(cx, cy)
        shape = (int(0.8 * box), int(0.8 * box))
        img = img[cx : img.shape[0] - cx, cy : img.shape[1] - cy, :]
        # print(img.shape)
        img = cv2.flip(img, 0)
        img = img.tobytes()
        imgSurf = pygame.image.frombuffer(img, shape, "RGB")
        imgSurf = pygame.transform.rotate(imgSurf, 180)
        dispX = int(2.25 * box)
        multiplier = 1.55

    disp = pygame.display.set_mode((dispX, int(1.1 * aspect[1])))
    manager = pygame_gui.UIManager((dispX, int(1.1 * aspect[1])))
    contrast_high = pygame_gui.elements.UIHorizontalSlider(
        relative_rect=pygame.Rect(
            (
                br * width * multiplier,
                int(aspect[1] // 10),
                int(aspect[1] * 0.25),
                int(aspect[1] * 1.1 // 20),
            )
        ),
        manager=manager,
        start_value=0.0,
        value_range=(100.0, 130.0),
    )
    contrast_low = pygame_gui.elements.UIHorizontalSlider(
        relative_rect=pygame.Rect(
            (
                br * width * multiplier,
                int(aspect[1] // 6),
                int(aspect[1] * 0.25),
                int(aspect[1] * 1.1 // 20),
            )
        ),
        manager=manager,
        start_value=0.0,
        value_range=(50.0, 80.0),
    )
    color_toggle = pygame_gui.elements.UIButton(
        relative_rect=pygame.Rect(
            (br * width * multiplier, aspect[1] // 3, aspect[1] // 4, aspect[1] // 20)
        ),
        text="Toggle Color",
        manager=manager,
    )
    quit_button = pygame_gui.elements.UIButton(
        relative_rect=pygame.Rect(
            (br * width * multiplier, aspect[1] // 1.2, aspect[1] // 4, aspect[1] // 20)
        ),
        text="Exit",
        manager=manager,
    )
    if not live:

        if is_restream:
            time_end = str(
                time_change(time_start, minutes=time_offset)
                if time_end is None
                else time_end
            )

            output = os.popen(
                f'restream.py -db {db} -dt sensor_raw -did {sensor} -ts "{time_start}" -te "{time_end}" -hid {hid} -stt True'
            ).readlines()

            stri = ""
            for x in output:
                stri += x
            print(stri)

            try:
                data_path = output[-2].replace("\n", "")
                open(data_path)
            except Exception as e:
                print(
                    "ERROR: Could not make API request. Sensor ID(sensor), Hive ID(hid), or Database(db) might need "
                    "to be changed"
                )
                quit()

            # restream_format(data_path)

        assert data_path is not None

        with open(data_path, "r") as f:
            text = f.readlines()
            if type(eval(text[0])) is tuple:
                text = text[0]

        if detection_path and not unified:
            with open(detection_path, "r") as f:
                detection_text = f.readlines()
            det_out_file = detection_text
        elif unified:
            det_out_file = text.copy()
        else:
            detection_text = None
            det_out_file = []
            times = {}

        playback = pygame_gui.elements.UIHorizontalSlider(
            relative_rect=pygame.Rect(
                (
                    box // 20,
                    int(aspect[1] * 1.02),
                    int(aspect_ratio[0] * 0.75),
                    int(aspect[1] * 1.1 // 20),
                )
            ),
            start_value=0.0,
            value_range=(0.0, 1.0),
            manager=manager,
        )
        text_line = 0
        if not unified:
            _thread.start_new_thread(
                stream_text_data, (text, det_out_file, fps, 0, np.inf)
            )
        else:
            _thread.start_new_thread(stream_unified_text, (text, fps))
        play_button = pygame_gui.elements.UIButton(
            relative_rect=pygame.Rect(
                (plb * width * 0.8, aspect[1] * 0.99, aspect[1] // 6, aspect[1] // 20)
            ),
            text="Play",
            manager=manager,
        )
        pause_button = pygame_gui.elements.UIButton(
            relative_rect=pygame.Rect(
                (
                    plb * width * 0.8,
                    aspect[1] * 1.0425 * 0.99,
                    aspect[1] // 6,
                    aspect[1] // 20,
                )
            ),
            text="Pause",
            manager=manager,
        )
        rw = pygame_gui.elements.UIButton(
            relative_rect=pygame.Rect(
                (
                    br * width * 0.75,
                    int(aspect[1] * 0.99),
                    aspect[1] // 10.75,
                    aspect[1] // 10.75,
                )
            ),
            text="<<",
            manager=manager,
        )
        ff = pygame_gui.elements.UIButton(
            relative_rect=pygame.Rect(
                (
                    (plb * width * 0.942),
                    aspect[1] * 0.99,
                    aspect[1] // 10.75,
                    aspect[1] // 10.75,
                )
            ),
            text=">>",
            manager=manager,
        )
        if label:
            if detection_path:
                edited_detection_path = detection_path[:-4] + "_EDITED.txt"
            else:
                edited_detection_path = data_path[:-4] + "_boundingbox_EDITED.txt"
            edit_button = pygame_gui.elements.UIButton(
                relative_rect=pygame.Rect(
                    (
                        br * width * multiplier,
                        aspect[1] // 2,
                        aspect[1] // 4,
                        aspect[1] // 20,
                    )
                ),
                text="Edit Bounding Box",
                manager=manager,
            )
            prev_button = pygame_gui.elements.UIButton(
                relative_rect=pygame.Rect(
                    (
                        br * width * 0.925 * multiplier,
                        aspect[1] // 1.8,
                        aspect[1] // 5,
                        aspect[1] // 20,
                    )
                ),
                text="Previous Frame",
                manager=manager,
            )
            next_button = pygame_gui.elements.UIButton(
                relative_rect=pygame.Rect(
                    (
                        br * width * 1.125 * multiplier,
                        aspect[1] // 1.8,
                        aspect[1] // 5,
                        aspect[1] // 20,
                    )
                ),
                text="Next Frame",
                manager=manager,
            )
            clear_button = pygame_gui.elements.UIButton(
                relative_rect=pygame.Rect(
                    (
                        br * width * multiplier,
                        aspect[1] // 1.635,
                        aspect[1] // 4,
                        aspect[1] // 20,
                    )
                ),
                text="Clear",
                manager=manager,
            )

            pos_drop = pygame_gui.elements.UIDropDownMenu(
                relative_rect=pygame.Rect(
                    (br * width, aspect[1] // 1.5, aspect[1] // 4, aspect[1] // 20)
                ),
                options_list=category_options,
                starting_option="Position",
                manager=manager,
            )

            pos_drop.visible = False
            clear_button.visible = False
            next_button.visible = False
            prev_button.visible = False
    else:
        _thread.start_new_thread(
            mqtt_processes, (mqtt_address, data_topic, None, "butlr", "2019Ted/")
        )
    try:
        logo = pygame.image.load("../logo/butlr.logo.png")
    except (FileNotFoundError, pygame.error):
        logo = pygame.image.load("logo/butlr.logo.png")
    disp.blit(logo, ((multiplier * width) - (box // 10), (1.1 * height) - (box // 10)))
    surf = pygame.surfarray.make_surface(
        color(np.full((8, 8), low_bound), low_bound, high_bound)
    )
    surf = pygame.transform.scale(surf, (box, box))

    contrast_text = title_font.render("Adjust Contrast", True, white)
    contrast_rect = contrast_text.get_rect()
    contrast_rect.topleft = (int(width * br) * multiplier, int(height // 20))
    try:
        time_text = font.render(
            f"Time: {datetime.fromtimestamp(timestamp)}", True, white
        )
    except:
        time_text = font.render(
            f"Time: {datetime.fromtimestamp(timestamp / 1000)}", True, white
        )
    time_rect = time_text.get_rect()
    time_rect.topleft = ((aspect_ratio[0] // 50) * multiplier, (aspect_ratio[1] // 50))
    low_bound_text = font.render("Low", True, white)
    low_bound_rect = low_bound_text.get_rect()
    low_bound_rect.topleft = (int(width * lb) * multiplier, int(height // 4.9))
    high_bound_text = font.render("High", True, white)
    high_bound_rect = high_bound_text.get_rect()
    high_bound_rect.topleft = (int(width * hb) * multiplier, int(height // 7.9))

    transparent_surface = pygame.Surface((box, box), pygame.SRCALPHA, 32)
    transparent_surface = transparent_surface.convert_alpha()


    if gtVidPath:
        frac = 0.8
        bsx = 20
        bsy = 2.5
    else:
        frac = 1
        bsx = 1
        bsy = 1

    while running:
        disp.fill(black)

        if label and labeling_active:
            clear_button.visible = True
            next_button.visible = True
            pos_drop.visible = True
            prev_button.visible = True
            pos_drop.show()
        elif label and not labeling_active:
            clear_button.visible = False
            next_button.visible = False
            prev_button.visible = False
            pos_drop.visible = False
            pos_drop.hide()

        if gtVidPath:
            disp.blit(imgSurf, (box // 20, box // 8))

        label_condiition = label and labeling_active and (not live)

        disp.blit(contrast_text, contrast_rect)
        disp.blit(low_bound_text, low_bound_rect)
        disp.blit(high_bound_text, high_bound_rect)
        offset = 1
        if multiplier == 1.55:
            offset = 3.025
        disp.blit(
            logo,
            (
                (multiplier * width) - ((offset * box) // (10)),
                (1.1 * height) - (box // 10),
            ),
        )

        for event in pygame.event.get():
            # QUIT
            if event.type == pygame.QUIT:
                running = False
            # ESC: quit this scene
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_SLASH:
                    ohno()
                    # mixer.quit()

                elif event.key == pygame.K_q and label_condiition:
                    pos_drop.kill()
                    pos_drop = pygame_gui.elements.UIDropDownMenu(
                        relative_rect=pygame.Rect(
                            (
                                br * width,
                                aspect[1] // 1.5,
                                aspect[1] // 4,
                                aspect[1] // 20,
                            )
                        ),
                        options_list=category_options,
                        starting_option="Sitting",
                        manager=manager,
                    )
                elif event.key == pygame.K_w and label_condiition:
                    pos_drop.kill()
                    pos_drop = pygame_gui.elements.UIDropDownMenu(
                        relative_rect=pygame.Rect(
                            (
                                br * width,
                                aspect[1] // 1.5,
                                aspect[1] // 4,
                                aspect[1] // 20,
                            )
                        ),
                        options_list=category_options,
                        starting_option="Standing",
                        manager=manager,
                    )
                elif event.key == pygame.K_e and label_condiition:
                    pos_drop.kill()
                    pos_drop = pygame_gui.elements.UIDropDownMenu(
                        relative_rect=pygame.Rect(
                            (
                                br * width,
                                aspect[1] // 1.5,
                                aspect[1] // 4,
                                aspect[1] // 20,
                            )
                        ),
                        options_list=category_options,
                        starting_option="Lying Down",
                        manager=manager,
                    )
                elif event.key == pygame.K_a and label_condiition:
                    pos_drop.kill()
                    pos_drop = pygame_gui.elements.UIDropDownMenu(
                        relative_rect=pygame.Rect(
                            (
                                br * width,
                                aspect[1] // 1.5,
                                aspect[1] // 4,
                                aspect[1] // 20,
                            )
                        ),
                        options_list=category_options,
                        starting_option="Uncertain",
                        manager=manager,
                    )
                elif event.key == pygame.K_f and label_condiition:
                    text_line = max(0, text_line - 1)
                elif event.key == pygame.K_g and label_condiition:
                    text_line += 1
                    if text_line == text_length:
                        text_line -= 1
                elif event.key == pygame.K_SPACE:
                    if not label_condiition:
                        pause = not pause
                elif event.key == pygame.K_r:
                    if not labeling_active:
                        pause = True
                        labeling_active = True
                        edit_button.set_text("Done Editing")
                    else:
                        labeling_active = False
                        edit_button.set_text("Edit Bounding Box")
                elif event.key == pygame.K_t:
                    if color_on:
                        color = lambda x, y, z: grayscale(x, y, z)
                        color_on = False
                    else:
                        color = lambda x, y, z: thermal(x, y, z)
                        color_on = True
                elif label_condiition and pygame.K_c:
                    if detection_path and not unified:
                        sensor = eval(det_out_file[det_curr])["ID"]
                        det_out_file[det_curr] = str(
                            {"bounding box": [], "timestamp": timestamp, "ID": sensor}
                        )
                        bb_lbl_dict[timestamp] = []

                    elif not detection_path and not unified:
                        times[timestamp] = []
                    else:
                        assert unified and not detection_path
                        cleared = eval(text[text_line])
                        cleared = {
                            k: v if k != "bbox" else [] for k, v in cleared.items()
                        }
                        text[text_line] = str(cleared) + "\n"

                    transparent_surface = pygame.Surface(
                        (box, box), pygame.SRCALPHA, 32
                    )
                    transparent_surface = transparent_surface.convert_alpha()
                    disp.blit(surf, (box // 20, box // 20))
                    disp.blit(transparent_surface, (box // 20, box // 20))
                    pygame.display.update()
                elif event.key == pygame.K_LEFT:
                    text_line = max(text_line - 100, 0)

                elif event.key == pygame.K_RIGHT:
                    try:
                        text_line = min(text_length - 1, text_line + 100)
                    except IndexError:
                        text_line = -1

            if event.type == pygame.MOUSEBUTTONDOWN:
                pos = pygame.mouse.get_pos()
                # print(det_curr)
                if label_condiition:
                    x = pos[0]
                    y = pos[1]
                    # print(x)
                    if multiplier > 1:
                        start = box - (box // 20)
                    else:
                        start = 0
                    if (
                        0 <= (x - (bsx * box // 20)) <= box / (frac * multiplier)
                        and 0 <= y - (bsy * box // 20) <= box
                    ):
                        if not snap:
                            points.append(
                                [
                                    np.around(
                                        (x - box // 20 - start) / (frac * box), 4
                                    ),
                                    np.around((y - bsy * box // 20) / (frac * box), 4),
                                ]
                            )
                        else:
                            if h == 8 or h == 32:
                                possible_coord = []
                                pos_co = 0
                                xx = x
                                for x in range(int(h) + 1):
                                    possible_coord.append(pos_co)
                                    pos_co += box / h
                                pt_x = xx - (box // 20) - start
                                pt_y = y - box // 20
                                pt_x = possible_coord[closest(possible_coord, pt_x)]
                                pt_y = possible_coord[closest(possible_coord, pt_y)]
                                pt = [
                                    np.around(pt_x / (frac * box), 4),
                                    np.around(pt_y / (frac * box), 4),
                                ]
                                points.append(pt)

                        if len(points) == 2:
                            # print(det_out_file)
                            new_boxes.append([p[::-1] for p in points])

                            for x in new_boxes:
                                bb_lbl.append(selected_pos)

                            if detection_path and not unified:
                                try:
                                    det_line = eval(det_out_file[det_curr])
                                    if det_line["timestamp"] == timestamp:
                                        current_packet = det_line
                                    else:
                                        raise Exception
                                except Exception:
                                    current_packet = {
                                        "bounding box": [],
                                        "timestamp": timestamp,
                                        "ID": sensor_mac,
                                    }
                                    det_out_file.insert(det_curr, str(current_packet))

                                bbs = current_packet["bounding box"]
                                # print("yeehaw")

                            elif not detection_path and not unified:
                                bbs = times[timestamp]

                            else:
                                assert not detection_path and unified
                                full_packet = text[text_line]
                                bbs = eval(full_packet)["bbox"]
                                boxes_out = []
                                for b in new_boxes:
                                    x_center = (b[1][0] + b[0][0]) * 0.5
                                    y_center = (b[1][1] + b[0][1]) * 0.5
                                    b_width = np.abs(b[1][0] - b[0][0])
                                    b_height = np.abs(b[1][1] - b[0][1])
                                    boxes_out.append(
                                        [x_center, y_center, b_width, b_height]
                                    )
                                new_boxes = boxes_out

                            try:
                                if len(bb_lbl_dict[timestamp]) >= 0:
                                    bb_lbl_dict[timestamp].extend(bb_lbl)
                                    # print(bb_lbl_dict[timestamp])
                                else:
                                    bb_lbl_dict[timestamp] = bb_lbl
                            except:
                                bb_lbl_dict[timestamp] = bb_lbl

                            bbs.extend(new_boxes)
                            bbs, bb_lbl_dict[timestamp] = (
                                list(t)
                                for t in zip(*sorted(zip(bbs, bb_lbl_dict[timestamp])))
                            )
                            # bbs.sort3(key=lambda p: p[0]) # sort based on x coordinate of first element
                            # print(bbs, bb_lbl)
                            # print([x for _, x in sorted(zip(bb_lbl, bbs), key=lambda pair: pair[0])])

                            if detection_path and not unified:
                                # temp_lbl = [-1 for x in bbs]
                                det_out_file[det_curr] = str(
                                    {
                                        "bounding box": bbs,
                                        "timestamp": timestamp,
                                        "ID": sensor_mac,
                                    }
                                )
                                # bb_lbl_dict[timestamp] = str(det_out_file[det_curr]["category_id"])
                                for det in det_out_file:
                                    eval(det)
                            elif not detection_path and not unified:
                                times[timestamp] = bbs
                                # bb_lbl_dict[timestamp] = bb_lbl

                            elif not detection_path and unified:
                                new_line = eval(text[text_line])
                                new_line["bbox"] = bbs
                                text[text_line] = str(new_line) + "\n"

                            # print(times)
                            # print(bb_lbl_dict)

                            points = []
                            bb_lbl = []
                            new_boxes = []

            if event.type == pygame.USEREVENT:
                if event.user_type == pygame_gui.UI_HORIZONTAL_SLIDER_MOVED:
                    if not live and event.ui_element == playback:
                        text_line = int(text_length * playback.get_current_value())
                        if gtVidPath:
                            c.set(1, int(total_frames * playback.get_current_value()))
                        playback.update(0.0001)
                        # print(playback.get_current_value())
                    elif event.ui_element == contrast_low:
                        pos = contrast_low.get_current_value()
                        low_bound = pos
                        contrast_low_text = font.render(str(int(pos)), True, white)
                        low_rect = contrast_low_text.get_rect()
                        low_rect.topleft = (
                            (aspect[0] - (box // 16)),
                            int(aspect[1] // 5.5),
                        )
                        disp.blit(contrast_low_text, low_rect)
                    elif event.ui_element == contrast_high:
                        pos = contrast_high.get_current_value()
                        high_bound = pos
                        contrast_high_text = font.render(str(int(pos)), True, white)
                        high_rect = contrast_high_text.get_rect()
                        high_rect.topleft = (
                            (aspect[0] - (box // 16)),
                            int(aspect[1] // 8.75),
                        )  # 8.75
                        disp.blit(contrast_high_text, high_rect)

                elif (
                    event.user_type == pygame_gui.UI_DROP_DOWN_MENU_CHANGED
                    and label_condiition
                ):
                    selected_pos = category_ids[
                        category_options.index(pos_drop.selected_option)
                    ]

                elif event.user_type == pygame_gui.UI_BUTTON_PRESSED:
                    if event.ui_element == color_toggle:
                        if color_on:
                            color = lambda x, y, z: grayscale(x, y, z)
                            color_on = False
                        else:
                            color = lambda x, y, z: thermal(x, y, z)
                            color_on = True
                    elif event.ui_element == quit_button:

                        if label:
                            if detection_path and not unified:
                                det_out_file = [
                                    x for x in det_out_file if eval(x)["bounding box"]
                                ]
                                out = []
                                for x in det_out_file:
                                    if eval(x)["bounding box"]:
                                        # print(eval(x))
                                        b = list(eval(x)["bounding box"])
                                        bb_out = []
                                        for b in b:
                                            timestamp = str(eval(x)["timestamp"])
                                            cx = (b[0][0] + b[1][0]) / 2
                                            cy = (b[0][1] + b[1][1]) / 2
                                            w = np.abs(b[1][0] - b[0][0])
                                            h = np.abs(b[1][1] - b[0][1])
                                            bb_out.append([cx, cy, w, h])

                                            try:
                                                if int(timestamp) in list(
                                                    bb_lbl_dict.keys()
                                                ):
                                                    cat_ids = bb_lbl_dict[
                                                        int(timestamp)
                                                    ]
                                                else:
                                                    cat_ids = []
                                            except:
                                                if float(timestamp) in list(
                                                    bb_lbl_dict.keys()
                                                ):
                                                    cat_ids = bb_lbl_dict[
                                                        float(timestamp)
                                                    ]
                                                else:
                                                    cat_ids = []
                                        out.append(
                                            str(
                                                json.dumps(
                                                    {
                                                        "image": data.tolist(),
                                                        "bbox": bb_out,
                                                        "category_id": cat_ids,
                                                        "timestamp": timestamp,
                                                        "mac_address": sensor_mac,
                                                        "normalized": norm,
                                                    }
                                                )
                                            )
                                        )
                                out.pop(-1)
                                with open(edited_detection_path, "w") as f:
                                    f.write("\n".join(out))

                            # Old format
                            # elif not detection_path and not unified:
                            #     det_out_file = [str({"bounding box": v, "timestamp": k, "ID": ""}) for k, v in times.items() if v]
                            #     with open(edited_detection_path, "a+") as f:
                            #         f.writelines([x + "\n" for x in det_out_file])

                            # New format
                            elif not detection_path and not unified:
                                det_out_file = [
                                    str({"bounding box": v, "timestamp": k, "ID": ""})
                                    for k, v in times.items()
                                    if v
                                ]
                                det_out_file = []

                                for ind in range(len(times.keys())):
                                    bb_out = []

                                    b = list(times.values())[ind]
                                    t = list(times.keys())[ind]
                                    try:
                                        p = list(bb_lbl_dict[t])
                                    except:
                                        p = []

                                    if b:
                                        bb_out = []
                                        b = list(b)
                                        for b in b:
                                            # b = list(b)[0]
                                            cx = (b[0][0] + b[1][0]) / 2
                                            cy = (b[0][1] + b[1][1]) / 2
                                            w = np.abs(b[1][0] - b[0][0])
                                            h = np.abs(b[1][1] - b[0][1])
                                            bb_out.append([cx, cy, w, h])

                                    det_out_file.append(
                                        str(
                                            json.dumps(
                                                {
                                                    "image": data.tolist(),
                                                    "bbox": bb_out,
                                                    "category_id": p,
                                                    "timestamp": t,
                                                    "mac_address": sensor_mac,
                                                    "normalized": norm,
                                                }
                                            )
                                        )
                                    )
                                with open(edited_detection_path, "a+") as f:
                                    f.writelines([x + "\n" for x in det_out_file])
                            else:
                                with open(data_path[:-4] + "_EDITED.txt", "w") as f:
                                    f.writelines(text)

                        running = False

                    elif label_condiition and event.ui_element == clear_button:
                        if detection_path and not unified:
                            sensor = eval(det_out_file[det_curr])["ID"]
                            det_out_file[det_curr] = str(
                                {
                                    "bounding box": [],
                                    "timestamp": timestamp,
                                    "ID": sensor,
                                }
                            )
                            bb_lbl_dict[timestamp] = []

                        elif not detection_path and not unified:
                            times[timestamp] = []
                            bb_lbl_dict[timestamp] = []

                        else:
                            assert unified and not detection_path
                            cleared = eval(text[text_line])
                            cleared = {
                                k: v if k != "bbox" else [] for k, v in cleared.items()
                            }
                            text[text_line] = str(cleared) + "\n"

                        transparent_surface = pygame.Surface(
                            (box, box), pygame.SRCALPHA, 32
                        )
                        transparent_surface = transparent_surface.convert_alpha()
                        disp.blit(surf, (box // 20, box // 20))
                        disp.blit(transparent_surface, (box // 20, box // 20))
                        pygame.display.update()

                    elif not live and event.ui_element == play_button:
                        if not label_condiition:
                            pause = False

                    elif not live and event.ui_element == pause_button:
                        pause = True
                    elif not live and event.ui_element == ff:
                        try:
                            try:
                                text_line = min(len(data_text) - 1, text_line + 100)
                            except:
                                text_line = min(text_length - 1, text_line + 100)
                        except IndexError:
                            text_line = -1
                    elif not live and event.ui_element == rw:
                        text_line = max(text_line - 100, 0)
                    elif not live and label and event.ui_element == edit_button:
                        if not labeling_active:
                            pause = True
                            labeling_active = True
                            edit_button.set_text("Done Editing")
                        else:
                            labeling_active = False
                            edit_button.set_text("Edit Bounding Box")
                    elif label_condiition and event.ui_element == next_button:
                        text_line += 1
                        if text_line == text_length:
                            text_line -= 1
                    elif label_condiition and event.ui_element == prev_button:
                        text_line = max(0, text_line - 1)
                    elif label_condiition and event.ui_element == clear_button:
                        if detection_path and not unified:
                            sensor = eval(det_out_file[det_curr])["ID"]
                            det_out_file[det_curr] = str(
                                {
                                    "bounding box": [],
                                    "timestamp": timestamp,
                                    "ID": sensor,
                                }
                            )
                        elif not detection_path and not unified:
                            times[timestamp] = []
                        else:
                            assert unified and not detection_path
                            cleared = eval(text[text_line])
                            cleared = {
                                k: v if k != "bbox" else [] for k, v in cleared.items()
                            }
                            text[text_line] = str(cleared) + "\n"
                        transparent_surface = pygame.Surface(
                            (box, box), pygame.SRCALPHA, 32
                        )
                        transparent_surface = transparent_surface.convert_alpha()
                        disp.blit(surf, (bsx * box // 20, bsy * box // 20))
                        disp.blit(
                            transparent_surface, (bsx * box // 20, bsy * box // 20)
                        )
                        pygame.display.update()

            manager.process_events(event)
        manager.update(0.0001)

        while len(data_queue) >= 1:
            packet = data_queue.popleft()
            if not unified:
                sensor_mac = packet["fields"]["macAddress"]
                if len(packet["fields"]["data"]) == 72:
                    data = packet["fields"]["data"][6:-2]
                    data = np.array(data)
                elif len(packet["fields"]["data"]) == 32:
                    if type(packet["fields"]["data"]) is list:
                        data = [[x * 4 for x in y] for y in packet["fields"]["data"]]
                    else:
                        data = packet["fields"]["data"] * 4
                elif len(packet["fields"]["data"]) == 74:
                    data = packet["fields"]["data"][6:-4]
                    data = np.array(data)
                else:
                    data = np.array(packet["fields"]["data"])
                try:
                    h = int(np.sqrt(data.size))
                except:
                    data = np.asarray(data)
                    h = int(np.sqrt(data.size))

                # print(h)

            else:
                data = packet
                data = np.asarray(data)
                h = int(np.sqrt(data.size))
                if h == 32:
                    data *= 4
            img = data
            data = data.reshape((h, h))
            if mac == "00-17-0d-00-00-70-b9-e3":
                data = data.T
            #     data = np.asarray([r[::-1] for r in data])

            surf = pygame.surfarray.make_surface(color(data, low_bound, high_bound))
            surf = pygame.transform.scale(surf, (int(frac * box), int(frac * box)))
            transparent_surface = pygame.Surface(
                (frac * box, frac * box), pygame.SRCALPHA, 32
            )
            transparent_surface = transparent_surface.convert_alpha()

            if gtVidPath:
                c.get(0)
                c.set(1, int(total_frames * playback.get_current_value()))
                success, img = c.read()
                shape = img.shape[1::-1]
                # print(img.shape)
                assert box < shape[0] and box < shape[1]
                cx = 120
                cy = 400
                # print(cx, cy)
                shape = (int(frac * box), int(frac * box))
                img = img[cx : img.shape[0] - cx, cy : img.shape[1] - cy, :]
                # print(img.shape)
                img = cv2.flip(img, 0)
                img = img.tobytes()
                # print(type(img))
                imgSurf = pygame.image.frombuffer(img, shape, "RGB")
                imgSurf = pygame.transform.rotate(imgSurf, 180)
                disp.blit(imgSurf, (box // 20, box // 8))

        disp.blit(surf, (bsx * box // 20, bsy * box // 20))
        while len(detection_queue) >= 1:
            # print(detection_queue)
            texts = []
            transparent_surface = pygame.Surface(
                (int(frac * box), int(frac * box)), pygame.SRCALPHA, 32
            )
            transparent_surface = transparent_surface.convert_alpha()
            boxes = detection_queue.popleft()
            if color_on:
                rect_color = black
            else:
                rect_color = red

            cnt = 0
            for b in boxes:
                z = aspect_ratio[0] * frac
                if not label:
                    try:
                        b = [z * k for k in b]
                        bb = pygame.Rect((b[0], b[1], b[2], b[3]))
                        pose = "Lying Down" if b[-1] else "Normal"
                        pose_text = title_font.render(pose, True, black)
                        text_rect = pose_text.get_rect()
                        text_rect.center = (
                            b[0] + (b[2] // 2) + (aspect_ratio[0] // 20),
                            b[1] - 10 + (aspect_ratio[1] // 20),
                        )
                        texts.append((pose_text, text_rect))
                    except:
                        b = [[z * k for k in d] for d in b]
                        box_height = np.abs(b[1][0] - b[0][0])
                        box_width = np.abs(b[1][1] - b[0][1])
                        bb = pygame.Rect((b[0][1], b[0][0], box_width, box_height))
                else:
                    b = [[z * k for k in d] for d in b]
                    box_height = np.abs(b[1][0] - b[0][0])
                    box_width = np.abs(b[1][1] - b[0][1])
                    bb = pygame.Rect((b[0][1], b[0][0], box_width, box_height))
                    try:
                        pose = f"{str(cnt)}-{category_options[category_ids.index(bb_lbl_dict[timestamp][cnt])]}"
                        pose_text = title_font.render(pose, True, black)
                        text_rect = pose_text.get_rect()
                        # text_rect.center = (b[0][0] + ((b[1][0]-b[0][0]) // 2) + (aspect_ratio[0] // 20), b[0][1] - 10 + (aspect_ratio[1] // 20))

                        texts.append((pose_text, bb))
                        cnt += 1
                    except:
                        pass
                pygame.draw.rect(transparent_surface, rect_color, bb, 2)

        disp.blit(transparent_surface, (bsx * box // 20, bsy * box // 20))
        for t in texts:
            disp.blit(t[0], t[1])

        manager.draw_ui(disp)
        if not live:
            current_pos = playback.get_current_value()
            if current_pos >= 0.99999:
                text_line = -1
                # exit(0)

        pygame.display.update()

        clock.tick(100)
        time.sleep(0.001)


def grayscale(im, lo=70, hi=110):
    im = 255 * ((im - lo) / (hi - lo))  # ((255 / (hi - lo)) * im) - lo
    w, h = im.shape
    ret = np.empty((w, h, 3), dtype=np.uint8)
    ret[:, :, 2] = ret[:, :, 1] = ret[:, :, 0] = im
    return ret


def thermal(im, lo=70, hi=110):
    # cold = hsl(240, 88, 30) # (0, 0, 80)
    # hot =  hsl(60, 88, 60) # (255, 255, 153)
    im = (im - lo) / (hi - lo)
    w, h = im.shape
    # print(im)
    h1 = (180 * im) + 60
    hue = h1 * 0.0027777777777
    # print(hue[0, 0])
    lum = 30 + (65 * im)
    sat = np.full((w, h), 0.88)
    ret = np.empty((w, h, 3), dtype=np.uint8)

    for j in range(h):
        for i in range(w):
            if h1[i, j] >= 240:
                hue[i, j] = max_hue
                lum[i, j] = 100
            r, g, b = colorsys.hls_to_rgb(hue[i, j], lum[i, j], 0.88)
            ret[i, j, 0] = r
            ret[i, j, 1] = g
            ret[i, j, 2] = b
    # print(r, g, b)
    return ret


def stream_text_data(text_full, det_text, fps, start_time, end_time):
    global playback
    global text_line
    global data_text
    global data_queue
    global detection_queue
    global det_packet_curr
    global det_index
    global det_curr
    global text_length
    global pause
    global category_ids
    global category_options
    global pos_drop
    global timestamp
    global selected_pos
    global times
    last_time = time.time()
    new_file = not det_text
    last_detection_line = 0
    parse_array = False

    while True:
        try:
            selected_pos = category_ids[
                category_options.index(pos_drop.selected_option)
            ]
        except:
            selected_pos = -1
        time_check = time.time()
        t1 = time.time()
        try:
            if parse_array:
                raise Exception
            text = eval(text_full)[0]
            text_length = len(text)
            if text[text_line][0] == "b":
                text[text_line] = text[text_line][1:]
            line_check = eval(text[text_line])
        except Exception as e:
            parse_array = True
            try:
                text = eval(text_full)
            except:
                text = text_full
            text_length = len(text)
            array = lambda x: np.asarray(x)
            try:
                if text[text_line][0] == "b":
                    text[text_line] = str(text[text_line][3:-3])
            except:
                pass
            if text_line >= text_length:
                text_line = text_length - 1
            try:
                line_check = eval(text[text_line])[0]
            except:
                line_check = eval(text[text_line])
        if type(line_check) is not dict:
            line_check = ast.literal_eval(line_check)
        if line_check["name"] == "notifData":
            realtime = line_check["timestamp"]
            try:
                timestamp = line_check["timestamp"]
            except KeyError:
                timestamp = line_check["fields"]["utcSecs"]
            id = line_check["fields"]["macAddress"]
        else:
            text_line += 1
            if text_line == text_length:
                text_line -= 1
            continue
        if not new_file:
            detections_current = det_text[last_detection_line:]
            det_index = 0
            try:
                time_curr = eval(detections_current[det_index])["timestamp"]
                while time_curr <= timestamp:
                    # print(timestamp, time_curr)
                    if time_curr == timestamp:
                        # print(det_text[det_curr])
                        detection_queue.append(
                            eval(detections_current[det_index])["bounding box"]
                        )
                        det_packet_curr = eval(detections_current[det_index])
                        det_curr = det_index
                        # print(det_packet_curr)
                        break
                    else:
                        det_index += 1
                        time_curr = eval(detections_current[det_index])["timestamp"]
            except IndexError:
                pass
        else:
            if timestamp not in times.keys():
                times[timestamp] = []
            detection_queue.append(times[timestamp])
            """det_index = 0
            det_curr = 0
            try:
                time_curr = eval(det_text[0])["timestamp"]
                # print(time_curr)
                while time_curr <= timestamp:
                    if time_curr == timestamp:
                        detection_queue.append(eval(det_text[det_curr])["bounding box"])
                        det_packet_curr = eval(det_text[det_curr])
                        det_curr = det_index
                        break
                    else:
                        det_index += 1
                        time_curr = eval(det_text[det_curr])["timestamp"]
                        if det_index >= len(det_text):
                            det_curr = det_index
                            break
            except IndexError as e:
                pass"""
        playback.set_current_value(text_line / len(text))
        playback.update(1 / 100)
        if realtime >= start_time:
            if time_check - last_time >= 1 / (4 * fps):
                data_queue.append(line_check)
                if realtime >= end_time:
                    raise e
            if not pause:
                text_line += 1
                if text_line == text_length:
                    text_line -= 1
        time.sleep(1 / fps)

        if text_line > len(text) - 1:
            text_line = len(text) - 1
            pause = True

        data_text = text


def stream_unified_text(text_lines, fps):
    global playback
    global text_line
    global data_queue
    global detection_queue
    global det_packet_curr
    global det_index
    global det_curr
    global text_length
    global pause
    global timestamp
    global times
    global bb_lbl_dict

    last_detection_line = 0
    text_length = len(text_lines)
    for x in range(text_length):
        text_lines[x] = text_lines[x].rstrip()

    # print(text_length)
    while True:
        try:
            text = text_lines[text_line]
        except IndexError:
            text = text_lines[-1]
            text_line = text_length - 1

        text = eval(text)
        timestamp = text["timestamp"]
        class_labels = text["category_id"]
        data = text["image"]
        boxes = text["bbox"]
        sensor = text["mac_address"]
        bb_lbl_dict[timestamp] = class_labels
        data_queue.append(data)
        out_boxes = [
            [
                [b[0] - (0.5 * b[2]), b[1] - (0.5 * b[3])],
                [b[0] + (0.5 * b[2]), b[1] + (0.5 * b[3])],
            ]
            for b in boxes
        ]
        detection_queue.append(out_boxes)

        playback.set_current_value(text_line / text_length)
        playback.update(1 / 100)
        if not pause:
            text_line += 1
        time.sleep(1 / fps)


        if text_line > text_length - 1:
            text_line = text_length - 1
            pause = True

        data_text = text


def mqtt_processes(address, topic_raw_in, topic_detect_in, usn, pw):
    global data_queue
    global detection_queue
    global client
    global bufferMQ
    global sensor_timer
    global det_buffer
    global timestamp
    global vizMac

    bufferMQ = []
    det_buffer = []
    sensor_timer = ""

    def on_subscribe(client, userdata, mid, granted_qos):
        print("Subscribed: " + str(mid) + " " + str(granted_qos))

    def on_message(client, userdata, msg):
        global bufferMQ
        global sensor_timer
        global det_buffer
        global in_count
        global out_count
        global timestamp
        try:
            # message = json.loads(msg.payload)
            array = lambda x: np.asarray(x)
            message = eval(msg.payload)
            # print(message)
            if type(message) is tuple:
                message = message[0]
            # GENERAL CONDITION
            if list(message.keys())[0] == "fields":
                if message["name"] == "notifData":
                    try:
                        timestamp = message["fields"]["timestamp"]
                    except:
                        timestamp = message["timestamp"]
                    if vizMac == message["fields"]["macAddress"]:
                        data_queue.append(message)
                        bbs = network(message)
                        if bbs:
                            detection_queue.append(bbs)

        except Exception as e:
            print(
                "\n========================================================================"
            )
            exc_type, exc_obj, exc_tb = sys.exc_info()
            print(
                "Error type: {}, happened on line {} in <mtqq_run>".format(
                    exc_type, exc_tb.tb_lineno
                )
            )
            print("Error: {}".format(e))
            print(
                "========================================================================\n"
            )

    client.username_pw_set(usn, pw)
    client.on_subscribe = on_subscribe
    client.on_message = on_message
    client.connect(address, 1883)
    client.subscribe(topic_raw_in, qos=1)
    if topic_detect_in:
        client.subscribe(topic_detect_in, qos=1)
    client.loop_forever()


def network(message):
    return []


def time_change(date, hours=0, minutes=0):
    return datetime.strptime(date, "%Y-%m-%d %H:%M:%S") + timedelta(
        hours=hours, minutes=minutes
    )


def date_format(date, format="%Y-%m-%d %H:%M:%S"):
    try:
        date_obj = datetime.strptime(date, format)
        return True
    except (ValueError, TypeError):
        return False


def ohno():
    mixer.init()
    mixer.music.set_volume(0.7)
    mixer.music.load("ohno.mp3")
    mixer.music.play()


def restream_format(path):
    with open(path, "r") as f:
        text = f.readlines()

    out_txt = "[["

    for x in text:
        x = x.rstrip()
        x = x.replace("'", '"')
        out_txt += "b'" + x + "', "

    out_txt = out_txt[:-2] + "]]"

    with open(path, "w") as f:
        f.write(out_txt)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-path", default="", help="Path to saved data (in .txt format)")
    parser.add_argument(
        "-det", default="", help="Path to saved detection/bounding box data"
    )
    parser.add_argument("-mqdi", default="", help="MQTT Data Channel")
    parser.add_argument(
        "-mqba",
        default="ec2-54-245-187-200.us-west-2.compute.amazonaws.com",
        help="MQTT Broker Address",
    )
    parser.add_argument("-fps", default="8")
    parser.add_argument("-sz", default="600", help="Size (in pixels) of data render.")
    parser.add_argument("-lbl", default="f")
    parser.add_argument("-mac", default="xxx")
    parser.add_argument("-uni", default="f")
    parser.add_argument("-gtv", default="")

    parser.add_argument("-restream", default="f")
    parser.add_argument("-ts", default="", help="Start time for restream")
    parser.add_argument("-te", default="", help="End time for restream")
    parser.add_argument(
        "-to", default="", help="Offset time if no end time for restream"
    )
    parser.add_argument("-hid", default="", help="Hive id for restream")
    parser.add_argument(
        "-sensor", default="", help="Sensor id(mac address)for restream"
    )
    parser.add_argument("-db", default="", help="Database for for restream")

    parser.add_argument(
        "-normalized", default="f", help="If the data is normalized or not"
    )
    parser.add_argument("-snap", default="f", help="Snap to nearest pixel")

    args = parser.parse_args()

    path = args.path if args.path else None
    det_path = args.det if args.det else None
    topic = args.mqdi if args.mqdi else None
    address = args.mqba if args.mqba else None
    fps = eval(args.fps)
    mac = args.mac
    unified = args.uni == "t"
    aspect_ratio = (eval(args.sz), eval(args.sz))

    try:
        live = (not path) and (len(topic) > 0) and (len(address) > 0)
    except:
        live = False

    label = args.lbl == "t"
    # for restream
    ts = args.ts if args.ts else None
    te = args.te if args.te else None
    to = args.to if args.to else None
    hid = args.hid if args.hid else None
    db = args.db if args.db else None
    sensor = args.sensor if args.sensor else None
    restream = args.restream == "t"

    normal = args.normalized == "t"
    snap = args.snap == "t"
    gtv = args.gtv

    visualizer(
        data_path=path,
        data_topic=topic,
        mqtt_address=address,
        fps=fps,
        aspect_ratio=aspect_ratio,
        live=live,
        label=label,
        detection_path=det_path,
        mac=mac,
        unified=unified,
        restream=restream,
        time_start=ts,
        time_offset=to,
        time_end=te,
        hid=hid,
        sensor=sensor,
        db=db,
        normalized=normal,
        snap=snap,
        gtVidPath=gtv,
    )

    # visualizer(data_path="test_data/lying_15_32x32_sensor.txt", data_topic="butlr/heatic/amg8833/test")
    # visualizer(mqtt_address="ec2-54-245-187-200.us-west-2.compute.amazonaws.com",
    #            data_topic="butlr/heatic/amg8833/test",
    #            live=True)


if __name__ == "__main__":
    main()
