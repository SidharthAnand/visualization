import os
import re
import ast
import sys
import json
import time
import pygame
import _thread
import argparse
import warnings
import pygame_gui
import numpy as np
from scipy import signal
from collections import deque
from datetime import datetime
import paho.mqtt.client as paho
from post_process import live_processing


def analyze(config_json_path=None, floorplan=None, address_dict=None, live=False, display='overlay', fps=10,
            aspect_ratio=(1000, 1000), grid_dimensions=None, data_path=None, det_path=None, post=True, convert='none',
            start_time=0, end_time=np.inf, show_graph=True, graph_path="", replaying=False, rotation=0,
            show_raw_detections=True, mqtt_address="ec2-34-222-201-1.us-west-2.compute.amazonaws.com", mqtt_usn="butlr",
            mqtt_pwd="2019Ted/", mqtt_in="butlr/idp_sante/heatic/raw_data", mqtt_out="butlr/idp_sante/heatic/out",
            mqtt_detections_in="butlr/idp_sante/heatic/detection",
            mqtt_detections_out="butlr/idp_sante/heatic/detections_post"):
    """
    :param config_json_path: (string) path to spatial configuration json file
    :param floorplan: (string) path to floorplan png file over which the sensor output will be shown
    :param address_dict: (dictionary) Should be made into a json argument to be loaded. For now, a dictionary of sensor
                         names and corresponding deviceNames
    :param live: (bool) Process live data
    :param display: (string) "grid"/"overlay"/"none" --> grid = no floorplan, overlay = floorplan, none = no viz
    :param fps: (int) frames per second for visualization
    :param aspect_ratio: (2-tuple) Aspect ratio of floorplan image in pixels, default (1000x1000)
    :param grid_dimensions: (3-tuple) (# plots in x direction, # plots in y direction, size of individual plot)
                            e.g. (6, 2, 100) would be a 2x6 grid of plots each of size 100x100px
    :param data_path: (string) path to saved raw data file
    :param det_path: (string) path to saved detection data file
    :param post: (bool) perform post processing
    :param convert: (str) 'wiwide', 'seamless', 'none'. Changes the way the data are parsed
    :param start_time: (int) Epoch mSec time from which to start analysis
    :param end_time: (int) Epoch mSec time to end analysis
    :param show_graph: (bool) show graph created for post processing
    :param graph_path: (str) path to graph (need not exist yet)
    :param replaying: (bool) data coming from Arihan's replay app. Formatting is a little different in this case.
    :param rotation: (float) degrees rotation counterclockwise (IDP --> -90, grid --> 90 or 0.)
    :param show_raw_detections: (bool) show raw detections and post-processed detections simultaneously
    :param mqtt_address: (string) address of broker
    :param mqtt_in: (string) raw data topic
    :param mqtt_out: (string) output topic (currently not used)
    :param mqtt_detections_in: (string) handcraft algorithm detection topic
    :param mqtt_usn: (string) MQTT username
    :param mqtt_pwd: (string) MQTT password
    :param mqtt_detections_out: (string) topic to which post-processed detections will be written
    :return: void
    """
    global data_queue
    global detection_queue
    global traj_queue
    global data_queue1
    global detection_queue1
    global post_queue
    global client

    global lo
    global hi

    global playback
    global text_line

    global show_det_local
    global use_delta

    lo = 100
    hi = 0

    os.environ['TZ'] = 'UTC'
    time.tzset()

    data_queue = deque([])
    detection_queue = deque([])
    traj_queue = deque([])

    data_queue1 = deque([])
    detection_queue1 = deque([])

    post_queue = deque([])
    client = paho.Client()

    wiwide = False
    seamless = False
    verizon = False
    if not replaying:
        if convert == 'wiwide':
            wiwide = True
        elif convert == 'seamless':
            seamless = True
        elif convert == 'verizon':
            verizon = True

    if config_json_path:
        addresses = {}
        centers = {}
        with open(config_json_path, "r") as cfg:
            full_json = json.load(cfg)
        config = full_json['sensors']
        coord_config = full_json['coordinates']
        exit_points = coord_config[0]['exits']
        try:
            rd_key = "room_dimensions"
            size_meters = full_json[rd_key]
        except KeyError:
            rd_key = "room dimensions"
            size_meters = full_json[rd_key]
        for s in config:
            addresses[s['label']] = s['deviceName']
            centers[s['label']] = s['center']
        print(addresses)
        if display == "overlay":
            aspect = aspect_ratio
            scaleX = aspect[0] / 1000
            scaleY = aspect[1] / 1000
            meterX = (aspect[0] / size_meters[0])
            meterY = (aspect[1] / size_meters[1])

            sizes, positions, rotations = _sizes_positions(full_json, meterX, meterY, aspect_ratio)

    if det_path and not os.path.exists(det_path) and not live:
        # data_converter(data_path, data_path[:-4] + "_converted.txt")
        converted = data_path
        sensitivity_params = "-cr2 0.4 -br2 0.8 -csh -0.8 -bsh -1.2 -csc 0.6 -bsc 0.6 -dr 3,2.5 -dmh both " \
                             "-dmc th -thh auto -thc -999 -thstdh 3.5 -thstdc 7.5 "
        sensitivity_params2 = "-cr2 0.3 -br2 0.5 -csh -0.5 -bsh -0.7 -csc 0.6 -bsc 0.6 -dmh both " \
                              "-dmc th -thh auto -thc -999 -thstdc 7.5 -thstdh 3"
        if config_json_path:
            config = f"-wldcd t -scfgp {config_json_path}"
        else:
            config = ""

        trk = "t" if wiwide else "f"

        for sensor in addresses.values():
            cmd = f'python3 Butlr_PoC.py -m saved_data -mqid {sensor} -dp {converted} -viz f -pub f -n t -amm t ' \
                  f'-mmdl 10 -imin 5 -imax 25 -wamm 1000 -famm 5 -abg t -rabg 5000 -fabg 10 -lt 100 -vt 10,50 ' \
                  f'-cr2 0.2 -br2 0.2 -csh -0.5 -bsh -0.6 -csc 0.6 -bsc 0.6 -dmh both -dmc th -dr 3,2.5 ' \
                  f'-thh auto -thc auto -thstdc 7.5 -thstdh 3 -ds 0.5001,0.0001 -de 0.5001,0.9999 -dprv 0 -ahr 0 ' \
                  f'-drt f -be f -tshc t -cfo t -wk t -art f -trk {trk} -dtcwf t {config} -dtcwp {det_path}'
            print("Running the handcraft algorithm on the provided dataset...")
            os.system(cmd)

    if display != 'none':
        pygame.init()

        white = (255, 255, 255)
        black = (0, 0, 0)
        red = (255, 0, 0)
        font = pygame.font.Font('font/Consola.ttf', 12)
        clock = pygame.time.Clock()

        if live:
            data_type = "live"
        else:
            data_type = "historical"

        if display == 'grid':
            if grid_dimensions is None:
                grid_dimensions = [int(np.around(len(addresses) / 2)), 2, 200]
            box = grid_dimensions[2]
            width = grid_dimensions[0] * box
            height = box * grid_dimensions[1]
            aspect = (width, height)
            font_size = min(12, int(grid_dimensions[2] // 25))

            font = pygame.font.Font('font/Consola.ttf', font_size)

            disp = pygame.display.set_mode((width, height + (font_size * 3)))
        else:
            if not live:
                disp = pygame.display.set_mode((aspect_ratio[0], int(1.1 * aspect_ratio[1])))
                manager = pygame_gui.UIManager((aspect_ratio[0], int(1.1 * aspect_ratio[1])))
                playback = pygame_gui.elements.UIHorizontalSlider(relative_rect=pygame.Rect((10,
                                                                                             int(aspect_ratio[1] * 1.02),
                                                                                             int(aspect_ratio[0] // 2),
                                                                                             int(aspect_ratio[
                                                                                                 1] * 1.1 // 20))),
                                                                  start_value=0.,
                                                                  value_range=(0., 1.),
                                                                  manager=manager)
            else:
                disp = pygame.display.set_mode(aspect)
        pygame.display.set_caption(f'butlr. {data_type} data analysis')

    sensors = list(addresses.keys())
    lo = 12
    hi = 45

    zero_matrix = np.zeros((8, 8))
    zero_matrix.fill(lo)

    if live:
        trajectory_topic = ""
        if wiwide:
            trajectory_topic = mqtt_in + "_out"
        _thread.start_new_thread(mqtt_processes, (mqtt_address, mqtt_in, mqtt_detections_in, trajectory_topic,
                                                  mqtt_detections_out, mqtt_usn, mqtt_pwd))

    else:
        det_start_idx = {}
        for sensor in sensors: det_start_idx[sensor] = 0
        if not os.path.exists(data_path[:-4] + "_sorted.txt"):
            print("Sorting data by timestamp")
            with open(data_path, "r") as f:
                unsorted0 = f.readlines()
            unsorted0.sort(key=lambda x: ast.literal_eval(x)['timestamp'])
            with open(data_path[:-4] + "_sorted.txt", "w+") as f:
                f.writelines(unsorted0)

        with open(data_path, "r") as f:
            text = f.readlines()
        if det_path:
            with open(det_path, "r") as f:
                det_text = f.readlines()
        text_line = 0

        post_path_out = ""
        if post:
            post_path_out = det_path[:-4] + "_POST.txt"
            _thread.start_new_thread(live_processing, (config_json_path, 3, None, detection_queue1,
                                                       post_path_out, False))

    matrices = {}
    trajectories = {}
    traj_times = {}
    trajectory_tracking = ""
    last_traj_sensor = ""
    last_detections = {}
    last_detections_local = {}
    world_detections = []
    objects = {}

    for ad in sensors:
        matrices[ad] = zero_matrix
        last_detections[ad] = []
        last_detections_local[ad] = []

    real_time_delta = 0
    epoch = 0
    det_time = 0
    last_time = time.time()
    flag = None
    buffer = []

    yeehaw = False
    qt1 = time.time()
    qt_last = 0

    running = True
    if display != "none":
        if not live:
            _thread.start_new_thread(stream_text_data, (text, wiwide, fps, sensors, start_time,
                                                        addresses, det_path, det_text, det_start_idx, post,
                                                        end_time, post_path_out))
        while running:
            for event in pygame.event.get():
                # QUIT
                if event.type == pygame.QUIT:
                    running = False
                # ESC: quit this scene
                if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                    running = False

                if event.type == pygame.MOUSEBUTTONDOWN:
                    pos = pygame.mouse.get_pos()
                    x = pos[0] / meterX
                    y = ((aspect[1] - pos[1]) / meterY)
                    """if display == 'overlay' and y > 0:
                        print("Pixels: ", pos)
                        print("Meters: ", (x, y))"""

                if event.type == pygame.USEREVENT and not live:
                    if event.user_type == pygame_gui.UI_HORIZONTAL_SLIDER_MOVED:
                        # print(event.ui_element, playback)
                        if event.ui_element == playback:
                            text_line = int(len(text) * playback.get_current_value())
                            playback.update(0.0001)
                            # print(playback.get_current_value())
                if not live: manager.process_events(event)
            if not live: manager.update(1/fps)

            while len(data_queue) > 0:
                mqtt_packet_list = data_queue.popleft()
                for mqtt_packet in mqtt_packet_list:
                    if (not wiwide or replaying) and "fields" in mqtt_packet.keys():
                        utc = mqtt_packet['fields']['utcSecs']
                        if not replaying:
                            if real_time_delta == 0:
                                real_time_delta = mqtt_packet['timestamp'] / 1000 - mqtt_packet['fields']['utcSecs']
                            data = mqtt_packet['fields']['data'][6:]
                        else:
                            if real_time_delta == 0 and use_delta:
                                real_time_delta = mqtt_packet['timestamp'] / 1000000 - mqtt_packet['fields']['utcSecs']
                            epoch /= 1000
                            data = mqtt_packet['fields']['data']
                        id = mqtt_packet['fields']['macAddress']
                        epoch = utc + real_time_delta
                    elif wiwide:
                        epoch = mqtt_packet['timestamp']
                        id = mqtt_packet['deviceName']
                        data = mqtt_packet['data']
                    if "flag" in mqtt_packet.keys():
                        flag = mqtt_packet['flag']

                    data = [min(max(x, lo * 4), hi * 4) for x in data]
                    if replaying: data = data[6:]
                    if seamless or verizon or replaying:
                        data = data[:-2]
                    if len(data) != 64:
                        print(f"improper data length: {len(data)} != 64")
                        raise Exception
                    data = np.asarray(data).reshape((8, 8)) * 0.25
                    sensor = None
                    for s in sensors:
                        if addresses[s] == id:
                            sensor = s
                    if sensor is None: continue
                    matrices[sensor] = np.array(data).T
                    last_detections[sensor] = []
                    last_detections_local[sensor] = []
                    world_detections = []

            while len(detection_queue) > 0:
                skip = False
                detect_packet_list = detection_queue.popleft()
                for detect_packet in detect_packet_list:
                    if not wiwide:
                        det_time = detect_packet['utcSecs'] + real_time_delta
                    else:
                        det_time = detect_packet['timestamp']
                    if replaying: det_time = detect_packet["timestamp"] / 1000
                    det_name = detect_packet['deviceName']
                    det_local = detect_packet['detectionsLocal']
                    det_world = detect_packet['detectionsWorld']

                    sensor = None
                    if not skip:
                        for s in sensors:
                            if addresses[s] == det_name:
                                sensor = s
                        if sensor is None: continue
                        last_detections_local[sensor] = det_local
                        last_detections[sensor] = det_world
                        if 'detectionsWorld' in detect_packet.keys():
                            world_detections.extend(detect_packet['detectionsWorld'])

            if len(traj_queue) > 0:
                in_n_out = {}
                trajectory = traj_queue.popleft()
                try:
                    traj_stamp = trajectory['deviceName']
                    trajectory_tracking = trajectory['timestamp']
                    traj_str = "trajectory"
                    in_n_out[traj_stamp] = (trajectory['in'], trajectory['out'])
                    # print(datetime.fromtimestamp(trajectory_tracking / 1000000))
                except KeyError:
                    traj_stamp = trajectory['DeviceName']
                    trajectory_tracking = trajectory['Time']
                    traj_str = "Trajectory"
                last_traj_sensor = f"(Sensor ID: {traj_stamp})"
                if traj_stamp not in trajectories.keys(): trajectories[traj_stamp] = []
                trajectories[traj_stamp].extend(trajectory[traj_str])
                traj_times[traj_stamp] = 0
            delete_times = []
            for s, t in traj_times.items():
                if t < 20:
                    traj_times[s] += 1
                else:
                    delete_times.append(s)
            for d in delete_times:
                del traj_times[d]
                del trajectories[d]

            while len(post_queue) > 0:
                objects_msg = post_queue.popleft()
                objects = {n: o for n, o in zip(objects_msg['detectionIDs'], objects_msg['detectionsWorld'])}
                ids = objects_msg['detectionIDs']

                qt2 = time.time()
                if int(qt2 - qt1) % 60 == 0 and not yeehaw:  # Ignore this variable name. I was having fun.
                    print(f"Length of post queue at {datetime.fromtimestamp(qt2)}: {len(post_queue)}")
                    yeehaw = True
                    qt_last = qt2
                if (time.time()) - qt_last > 10: yeehaw = False
                # det_time = objects_msg['timestamp'] / 1000
                # print(len(post_queue))

            if display == 'overlay':
                world_cd_surf = pygame.Surface(aspect, pygame.SRCALPHA, 32)
                world_cd_surf = world_cd_surf.convert_alpha()
                if show_graph:
                    with open(graph_path, "r") as f:
                        graph = json.load(f)
                    with open(config_json_path, "r") as f:
                        cf = json.load(f)
                    walls = cf['coordinates'][0]['walls']
                    for wall in walls:
                        pt0 = [wall[0][0] * meterX, aspect_ratio[1] - (wall[0][1] * meterY)]
                        pt1 = [wall[1][0] * meterX, aspect_ratio[1] - (wall[1][1] * meterY)]
                        # print(pt0, pt1)
                        pygame.draw.line(world_cd_surf, black, pt0, pt1, 1)
                    for name, v in graph.items():
                        node = v[0]
                        pos = node['position']
                        nbs = node['neighbors']
                        pygame.draw.circle(world_cd_surf, black, (pos[0] * meterX, pos[1] * meterY), 2, width=0)
                        for n in nbs:
                            nb_pos = graph[str(n[0])][0]['position']
                            pygame.draw.line(world_cd_surf, red, (pos[0] * meterX, aspect_ratio[1] - pos[1] * meterY),
                                             (nb_pos[0] * meterX, aspect_ratio[1] - nb_pos[1] * meterY), 1)

                disp.blit(world_cd_surf, (0, 0))

                assert floorplan is not None, "A floor plan image file must be provided for the overlay visualization mode."

                for pt in exit_points:
                    cd0 = int(pt[0] * meterX)
                    cd1 = aspect_ratio[1] - int(pt[1] * meterY)
                    pygame.draw.circle(world_cd_surf, black, (cd0, cd1), 8, width=1)
                img = pygame.image.load(floorplan)
                img = pygame.transform.scale(img, aspect)
                disp.blit(img, (0, 0))
                texts = {}
                for sensor, data in matrices.items():
                    data = np.clip(data, lo + 0.01, hi - 0.01).copy()
                    surf = pygame.surfarray.make_surface(gray(data, lo, hi))
                    surf = pygame.transform.scale(surf, sizes[sensor])
                    if objects:
                        # print(objects)
                        for pair in objects.values():
                            cd0 = int((pair[0]) * meterX)
                            cd1 = aspect_ratio[1] - int((pair[1]) * meterY)
                            pygame.draw.circle(world_cd_surf, (0, 0, 255), (cd0, cd1), 10, width=1)
                    if show_raw_detections:
                        if show_det_local:
                            for loc in last_detections_local[sensor]:
                                cd0 = int(loc[0] * 7 * (sizes[sensor][0] / 8))
                                cd1 = int(loc[1] * 7 * (sizes[sensor][1] / 8))
                                pygame.draw.circle(surf, (255, 255, 0), (cd0, cd1), 10, width=1)
                        for det in last_detections[sensor]:
                            cd0 = int(det[0] * meterX)
                            cd1 = aspect_ratio[1] - int(det[1] * meterY)
                            pygame.draw.circle(world_cd_surf, red, (cd0, cd1), 8, width=1)
                    surf = pygame.transform.rotate(surf, rotations[sensor])
                    disp.blit(surf, positions[sensor])
                    disp.blit(world_cd_surf, (0, 0))

                if post:
                    obs = font.render("OBJECTS:", True, black)
                    obRect = obs.get_rect()
                    obRect.topleft = (0, int(64 * scaleX))
                    disp.blit(obs, obRect)
                    start = 80
                    gap = 20
                    for k in range(len(objects)):
                        cds = list(objects.values())[k]
                        cd0 = np.around(cds[0], 2)
                        cd1 = np.around(cds[1], 2)
                        place = start + (k * gap)
                        texts[k] = font.render(f"Name: {ids[k]}, Coordinates: {(cd0, cd1)}",
                                               True, black)
                        textRect = texts[k].get_rect()
                        textRect.topleft = (0, place * scaleX)
                        disp.blit(texts[k], textRect)

                for text in texts.values(): text.fill(white)
                # plot last massage time
                if not replaying:
                    nt = epoch
                else:
                    nt = epoch * 1000
                try:
                    stamp = datetime.fromtimestamp(nt)
                except ValueError:
                    stamp = datetime.fromtimestamp(epoch)
                text0 = font.render(f'data time (GMT): {stamp}', True, black)
                textRect = text0.get_rect()
                textRect.topleft = (0, int(10 * scaleY))
                disp.blit(text0, textRect)

                # plot current time
                stamp = datetime.fromtimestamp(time.time())
                text1 = font.render(f'current time (GMT): {stamp}', True, black)
                textRect = text1.get_rect()
                textRect.topleft = (0, int(37 * scaleY))
                disp.blit(text1, textRect)

                # Plot last message time
                stamp = datetime.fromtimestamp(det_time)
                text2 = font.render(f'last detection time (GMT): {stamp}', True, black)
                textRect = text2.get_rect()
                textRect.topright = (aspect[0], int(10 * scaleY))
                disp.blit(text2, textRect)

            elif display == 'grid':
                transparent_surface = pygame.Surface(aspect, pygame.SRCALPHA, 32)
                transparent_surface = transparent_surface.convert_alpha()
                if post: warnings.warn("Post-processed detections will not be shown in the grid visualization mode")
                if not trajectories: disp.fill(black)
                for idx, tup in enumerate(matrices.items()):
                    deviceName = tup[0]
                    name = deviceName
                    data = tup[1]
                    surf = pygame.surfarray.make_surface(gray(data, lo, hi))

                    surf = pygame.transform.scale(surf, (box, box))
                    if verizon: surf = pygame.transform.rotate(surf, rotation)
                    x = box * (idx % grid_dimensions[0])
                    y = box * (idx // grid_dimensions[0])

                    pygame.draw.rect(transparent_surface, black, (x, y, box, box), width=1)
                    if type(deviceName) is int: name = addresses[deviceName]
                    text1 = font.render(f'{name}', True, black)
                    textRect = text1.get_rect()
                    textRect.topleft = (x + 3, y + 3)

                    for cd in last_detections[deviceName]:
                        cd0 = int(cd[0] * 7 * (grid_dimensions[2] / 8))
                        cd1 = int(cd[1] * 7 * (grid_dimensions[2] / 8))
                        pygame.draw.circle(surf, red, (cd0, cd1), box // 10, width=1)
                        # pygame.draw.circle(surf, red, cd, box // 10, width=1)

                    disp.blit(surf, (x, y))
                    disp.blit(text1, textRect)

                    if flag is not None:
                        flag_text = font.render(f"Flag: {flag}", True, black)
                        flagRect = flag_text.get_rect()
                        flagRect.topleft = (x + 3, y + 3 + font_size)
                        disp.blit(flag_text, flagRect)

                    if trajectories:
                        for k, traj in trajectories.items():
                            color = [0, 0, 0]
                            if in_n_out[k][0]:
                                color[0] += 255
                            if in_n_out[k][1]:
                                color[2] += 255
                            color = tuple(color)
                            try:
                                key = [idx for idx, x in enumerate(addresses.keys()) if addresses[x] == k][0]
                            except IndexError:
                                continue
                            move_y = key // grid_dimensions[0]
                            move_x = key % grid_dimensions[0]
                            for j in range(len(traj)):
                                trajectory = traj[j]
                                for i in range(len(trajectory) - 1):
                                    p0 = (trajectory[i][0] * box + (box * move_x),
                                          trajectory[i][1] * box + (box * move_y))
                                    p1 = (trajectory[i + 1][0] * box + (box * move_x),
                                          trajectory[i + 1][1] * box + (box * move_y))
                                    pygame.draw.circle(transparent_surface, color, p0, 4, width=0)
                                    pygame.draw.line(transparent_surface, color, p0, p1, width=1)
                                    if i == len(trajectory) - 1:
                                        pygame.draw.circle(transparent_surface, color, p1, 4, width=0)

                    disp.blit(transparent_surface, (0, 0))
                if wiwide:
                    try:
                        trajectory_tracking = datetime.fromtimestamp(int(trajectory_tracking) / 1000000)
                    except:
                        pass
                    traj_text = font.render(f"Last Trajectory: {trajectory_tracking} {last_traj_sensor}",
                                            True, white, black)
                    traj_rect = traj_text.get_rect()
                    traj_rect.topleft = (0, height + 5)
                    disp.blit(traj_text, traj_rect)

                data_time_text = font.render(f"Data Time: {datetime.fromtimestamp(epoch)}", True, white, black)
                dt_rect = data_time_text.get_rect()
                dt_rect.topleft = (0, height + 5 + font_size)

                disp.blit(data_time_text, dt_rect)

            if not live:
                manager.draw_ui(disp)
                current_pos = playback.get_current_value()
                if current_pos == 1: exit(0)
            pygame.display.update()

            clock.tick(100)
            time.sleep(0.001)


def stream_text_data(text, wiwide, fps, sensors, start_time,
                     addresses, det_path, det_text, det_start_idx,
                     post, end_time, post_path_out):
    global playback
    global text_line
    global data_queue
    last_time = time.time()
    buffer = []
    while True:
        time_check = time.time()
        if text[text_line][0] == "b": text[text_line] = text[text_line][1:]
        line_check = eval(text[text_line])
        if type(line_check) is not dict: line_check = ast.literal_eval(line_check)
        if not wiwide and line_check['name'] == "notifData":
            realtime = line_check['timestamp']
            timestamp = line_check['fields']['utcSecs']
            id = line_check['fields']['macAddress']
        elif wiwide:
            timestamp = line_check['timestamp']
            realtime = timestamp
            id = line_check['deviceName']
        else:
            text_line += 1
            continue
        sensor = None
        playback.set_current_value(text_line / len(text))
        playback.update(1 / fps)
        if realtime >= start_time:
            if time_check - last_time >= 1 / (4 * fps * len(sensors)):
                last_time = time_check
                if len(buffer) > 1:
                    data_queue.append(buffer)
                    buffer = []

                for s in sensors:
                    if addresses[s] == id:
                        sensor = s
                buffer.append(line_check)
                if det_path and sensor is not None:
                    for t in range(det_start_idx[sensor], len(det_text)):
                        packet = ast.literal_eval(det_text[t])
                        if wiwide:
                            time_unit = 'timestamp'
                        else:
                            time_unit = 'utcSecs'
                        if packet['deviceName'] == id and packet[time_unit] == timestamp:
                            if post: detection_queue1.append(packet)
                            detection_queue.append([packet])
                            det_start_idx[sensor] = t
                        elif packet[time_unit] > timestamp:
                            break
                elif realtime >= end_time:
                    raise
            text_line += 1

        if post and os.path.exists(post_path_out):
            with open(post_path_out, "r") as post_file:
                post_text = post_file.readlines()
            for line in post_text:
                post_packet = eval(line)
                post_timestamp = post_packet['timestamp']
                if post_timestamp == timestamp:
                    post_queue.append(post_packet)
        time.sleep(1 / (10*(fps * len(sensors))))


def _sizes_positions(config, meterX, meterY, aspect):
    cfg = config['sensors']
    positions = {}
    sizes = {}
    rotations = {}
    for sensor_dict in cfg:
        label = sensor_dict['label']

        center = sensor_dict['center']
        c0 = (center[0]) * meterX
        c1 = aspect[1] - ((center[1]) * meterY)

        dims = sensor_dict['coverage_dim']
        d0 = dims[0] * meterX
        d1 = dims[1] * meterY

        sizes[label] = (int(d0), int(d1))
        positions[label] = (int(c0 - (d0 / 2)), int(c1 - (d1 / 2)))
        try:
            rotations[label] = sensor_dict['rotation']
        except KeyError:
            rotations[label] = 0
    return sizes, positions, rotations


def mqtt_processes(address, topic_raw_in, topic_detect_in, traj_topic, topic_post_in, usn, pw):
    global data_queue
    global detection_queue
    global traj_queue
    global post_queue
    global client
    global bufferMQ
    global sensor_timer
    global det_buffer
    global in_count
    global out_count
    in_count = 0
    out_count = 0
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
        try:
            # message = json.loads(msg.payload)
            message = eval(msg.payload)

            # GENERAL CONDITION
            if list(message.keys())[0] == 'fields':
                if message['name'] == 'notifData':
                    bufferMQ.append(message)
                    # print("BUFFER:", len(bufferMQ))
                    if not sensor_timer:
                        sensor_timer = message['fields']['macAddress']
                    if message['fields']['macAddress'] == sensor_timer or len(bufferMQ) > 1:
                        data_queue.append(bufferMQ)
                        # print("QUEUE:", len(data_queue))
                        bufferMQ = []
            # WIWIDE CONDITION
            if list(message.keys())[0] == "flag":
                data_queue.append([message])

            elif "lostIDs" in message.keys():
                # print("post detection")
                post_queue.append(message)

            elif any([x in message.keys() for x in ['deviceName', 'DeviceName', 'detectionsLocal']]):
                if any([p in message.keys() for p in ['trajectory', "Trajectory"]]):
                    traj_queue.append(message)
                    in_count += message['in']
                    out_count += message['out']
                    print(f"TOTAL IN: {in_count} \nTOTAL OUT: {out_count}\n")
                else:
                    if len(message['detectionsLocal']) > 0:
                        detection_queue.append([message])
                        detection_queue1.append([message])

        except Exception as e:
            print('\n========================================================================')
            exc_type, exc_obj, exc_tb = sys.exc_info()
            print('Error type: {}, happened on line {} in <mtqq_run>'.format(exc_type, exc_tb.tb_lineno))
            print('Error: {}'.format(e))
            print('========================================================================\n')

    client.username_pw_set(usn, pw)
    client.on_subscribe = on_subscribe
    client.on_message = on_message
    client.connect(address, 1883)
    client.subscribe(topic_raw_in, qos=1)
    if topic_detect_in: client.subscribe(topic_detect_in, qos=1)
    if traj_topic: client.subscribe(traj_topic, qos=1)
    if topic_post_in: client.subscribe(topic_post_in, qos=1)
    client.loop_forever()


def sensor_count(path):
    sensor_ids = []
    with open(path, 'r') as f:
        text = f.readlines()
    for i in range(len(text)):
        try:
            if text[i][0] == "b": text[i] = text[i][1:]
            input_dict = ast.literal_eval(text[i])
            if type(input_dict) is not dict: input_dict = ast.literal_eval(input_dict)
            if "fields" in input_dict.keys():
                name = input_dict['fields']['macAddress']
            else:
                name = input_dict['deviceName']
            if name not in sensor_ids:
                sensor_ids.append(name)
        except Exception as e:
            print(e)
            continue
    return sensor_ids


def separate_sensors(path, sensor_ids):
    sections = {}
    with open(path, 'r') as f:
        text = f.readlines()

    for line in text:
        d = ast.literal_eval(line)
        name = d['deviceName']
        if name not in sections.keys():
            sections[name] = []
        sections[name].append(line)

    paths = {}
    for sensor in sensor_ids:
        with open(path[:-4] + '_' + sensor + ".txt", 'w') as f:
            f.writelines(sections[sensor])
        paths[sensor] = path[:-4] + '_' + sensor + ".txt"
    return paths


def data_converter(inputfile: str, outputfile: str):
    with open(inputfile, 'r') as f:
        lines = f.readlines()
    with open(outputfile, 'w') as f:
        for line in lines:
            if line[0] == "b": line = line[1:]
            line = ast.literal_eval(line)
            if type(line) is not str:
                line = str(line)
            data: dict = eval(line)
            fields: dict | int = data.get('fields', 0)
            if fields != 0:
                data_inside = fields.get('data', 0)
                if data_inside != 0:
                    json.dump({'data': data_inside[6:70],
                               'deviceName': fields['macAddress'],
                               'thermistor': int.from_bytes(data_inside[4:6], 'little'),
                               'timestamp': data['timestamp'],
                               'utcUsecs': data['fields']['utcUsecs'],
                               'utcSecs': data['fields']['utcSecs']}, f)
                    f.write('\n')


def gray(im, lo, hi):
    lo_pot = np.percentile(im, 2) - 3
    hi_pot = np.percentile(im, 99) + 7
    if lo_pot < lo:
        lo = lo_pot
    if hi_pot > hi:
        hi = hi_pot
    im = ((255 / (hi - lo)) * im) - lo
    w, h = im.shape
    ret = np.empty((w, h, 3), dtype=np.uint8)
    ret[:, :, 2] = ret[:, :, 1] = ret[:, :, 0] = im
    return ret


def bw(surf, arr):
    width, height = surf.get_size()
    for i in range(width):
        for j in range(height):
            if arr[i, j]:
                surf.set_at((i, j), (255, 255, 255))
    return surf


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-config", default="")
    parser.add_argument("-fp", default="")

    parser.add_argument("-ad", default="None")
    parser.add_argument("-live", default="f")

    parser.add_argument("-disp", default="grid")
    parser.add_argument("-ar", default="(1000, 1000)")
    parser.add_argument("-gd", default="None")
    parser.add_argument("-fps", default="10")
    parser.add_argument("-start", default="0")
    parser.add_argument("-end", default="np.inf")

    parser.add_argument("-data", default="")
    parser.add_argument("-det", default="")
    parser.add_argument("-post", default="f")
    parser.add_argument("-cvt", default="f")

    parser.add_argument("-shgr", default="f")
    parser.add_argument("-gp", default="")

    parser.add_argument('-mqad', default="")
    parser.add_argument('-mqus', default="")
    parser.add_argument('-mqpw', default="")
    parser.add_argument('-mqri', default="")
    parser.add_argument('-mqop', default="")
    parser.add_argument('-mqdi', default="")
    parser.add_argument('-mqdo', default="")

    parser.add_argument("-rp", default="f")
    parser.add_argument("-rot", default="0")
    parser.add_argument("-srd", default="t")

    parser.add_argument("-sld", default="f")
    parser.add_argument("-delta", default="t")

    args = parser.parse_args()

    global show_det_local
    global use_delta
    use_delta = args.delta == "t"
    show_det_local = args.sld == "t"

    config = args.config
    floorplan = args.fp

    address_dict = eval(args.ad)
    live = (args.live == "t")

    disp = args.disp
    aspect = eval(args.ar)
    grid_dims = eval(args.gd)
    fps = eval(args.fps)
    start = eval(args.start)
    end = eval(args.end)

    data_path = args.data
    det_path = args.det
    post = (args.post == "t")
    convert = args.cvt

    show_graph = args.shgr == "t"
    graph_path = args.gp

    replay = args.rp == "t"
    rotation = eval(args.rot)
    srd = args.srd == "t"

    mqad = args.mqad
    mqus = args.mqus
    mqpw = args.mqpw
    mqri = args.mqri
    mqdi = args.mqdi
    mqdo = args.mqdo
    mqop = args.mqop

    analyze(config_json_path=config, floorplan=floorplan, address_dict=address_dict, live=live, start_time=start,
            end_time=end, display=disp, aspect_ratio=aspect, grid_dimensions=grid_dims, data_path=data_path, fps=fps,
            replaying=replay, det_path=det_path, post=post, convert=convert, mqtt_address=mqad, show_graph=show_graph,
            graph_path=graph_path, mqtt_usn=mqus, mqtt_pwd=mqpw, mqtt_in=mqri, rotation=rotation,
            mqtt_detections_in=mqdi, mqtt_detections_out=mqdo, mqtt_out=mqop, show_raw_detections=srd)


if __name__ == "__main__":
    main()
