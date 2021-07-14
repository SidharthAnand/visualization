from shortest_paths import Pathfinder, nearest
import paho.mqtt.client as paho
from datetime import datetime
from collections import deque
import numpy as np
import argparse
import _thread
import json
import time
import sys
import re
import os


def live_processing(config_json, tolerance, mqtt_objects_out=None, general=None,
                    client=None, publish_to_mqtt=True, graph_path_out=None, txt_file_path_out=None):
    """
    :param config_json: (.json file) spatial configuration file detailing all of the specifications of the room
    :param tolerance: (float) radius in meters of the maximum distance we will allow people to move between frames
    :param mqtt_objects_out: (str) MQTT topic to publish results
    :param general: (Deque object or None) No need to worry about this unless changing analysis_general.py
    :param client: (paho-mqtt client or None) No need to worry about this unless changing analysis_general.py
    :param publish_to_mqtt: (bool) Generally True
    :param graph_path_out: (str or None) Path to graph of space. If a path is given but does not exist,
                            a graph will automatically be created and written to the specified path. If a path is not
                            provided, Euclidean Distance will be used to calculate distances between points instead of
                            the A* Shortest Paths Algorithm.
    :param txt_file_path_out: (str) If not publishing to MQTT, this parameter must be specified so the results
                                will be written to a text file.
    :return: void
    """
    global objects
    global potential_objects
    global object_id
    global mqtt_out_dict
    global detection_queue
    global post_queue

    global exit_tol_scale
    global minimum_to_register

    if general is not None:
        detection_queue = general
    else:
        detection_queue = deque([])
    post_queue = deque([])

    mqtt_out_dict = {}
    potential_objects = []
    potential_update_count = {}
    objects = {}
    object_id = 0
    last_send_time = 0
    last_detection_time = 0

    # Spatial configuration and graph creation
    if graph_path_out == '':
        graph_path_out = None
    with open(config_json, "r") as cfg:
        full_json = json.load(cfg)
    config = full_json['sensors']
    deviceNames = [x['deviceName'] for x in config]
    coordinate_config = full_json['coordinates'][0]
    exit_coordinates = coordinate_config['exits']
    try:
        exits_tolerance = coordinate_config['exits_tolerance']
    except KeyError:
        exits_tolerance = tolerance

    if graph_path_out is not None and not os.path.exists(graph_path_out):
        print("Discretizing the environment...")
        make_graph(config_json, graph_path_out)
        time.sleep(3)

    if graph_path_out:
        print("Calculating distances with A* algorithm for shortest paths.")
        pf = Pathfinder(graph_path_out, a_star=False)
        dist = lambda pt1, pt2: pf.dijkstra(nearest(pt1, pf.nodes), nearest(pt2, pf.nodes))
    else:
        dist = lambda pt1, pt2: euclidean_distance(pt1, pt2)
        print("Calculating distances with euclidean distance formula.")

    lost_ids = []
    yeehaw = False

    qt1 = time.time()
    qt_last = 0
    while True:
        try:
            object_id = max(list(objects.keys())) + 1
        except ValueError:
            object_id = 0

        # If we have detections registered from the handcraft algorithm...
        seen = []
        while len(detection_queue) > 0:
            # Check the length of the detection queue every 60 seconds to make sure we are not getting backlogged
            qt2 = time.time()
            if int(qt2 - qt1) % 60 == 0 and not yeehaw:     # Ignore this variable name. I was having fun.
                # print(f"Length of detection queue at {datetime.fromtimestamp(qt2)}: {len(detection_queue)}")
                yeehaw = True
                qt_last = qt2
            if (time.time()) - qt_last > 10: yeehaw = False

            # Data extraction
            det_list = detection_queue.popleft()
            det_sensor = []
            det_local = []
            detections = []
            local_detections = []

            # Put all aggregated frames into one "super message" so that we can take detections from many
            # sensors at once. We then check whether the detections we've collected are near the edges of the field of
            # view of their respective sensors and collapse duplicate detections which result from overlaps in sensor
            # coverage into a single detection.
            timestamp = [packet['timestamp'] for packet in det_list][-1]
            if not last_detection_time: last_detection_time = timestamp
            for k in range(len(det_list)):
                packet = det_list[k]
                detections.extend(packet['detectionsWorld'])
                local_detections.extend(packet['detectionsLocal'])
                det_sensor.append(packet['deviceName'])
            detections = check_edges(local_detections, detections, det_sensor)
            try:
                det_utc = [det_packet['utcSecs'] for det_packet in det_list][0]
                det_usec = [det_packet['utcUsecs'] for det_packet in det_list][0]
            except KeyError:
                continue

            # Check the potential objects against the new detections to see if they should be actualized.
            # If the distance is less than the specified tolerance, we can consider the potential
            # object to be a real person and add him to the object list.
            for p in potential_objects:
                if not any([dist(p, k) < tolerance / 2 for k in potential_update_count.keys()]):
                    potential_update_count[tuple(p)] = [0, 0]

            # New potential object actualization protocol:
            # Assert that the potential objects must be updated for 10 (not necessarily consecutive) frames in
            # a 50 frame period before being actualized so that we are not accidentally creating objects from noise.
            for d in [x for x in detections if not any([dist(ob, x) < tolerance / 2 for ob in objects.values()])]:
                for k, v in potential_update_count.items():
                    if minimum_to_register < euclidean_distance(d, k) < tolerance:
                        potential_update_count.pop(k)
                        potential_update_count[tuple(d)] = [v[0] + 1, v[1]]
                        # print(potential_update_count)
                        break
            potential_update_count = {k: [v[0], v[1] + 1] for k, v in potential_update_count.items() if v[1] < 50}
            used = []
            updated = []
            potentials = {k: v for k, v in potential_update_count.items() if v[0] >= 10}
            for p, c in potentials.items():
                objects[object_id] = p
                # print("Potential object actualized:", objects)
                updated.append(object_id)
                seen.append(object_id)
                object_id = max(list(objects.keys())) + 1
                used.append(p)

            potential_objects = []
            for u in used:
                potential_update_count.pop(u)

            # This case checks whether or not we should update an existing list of objects.
            # If the new detection coordinates are within a certain radius of an existing point,
            # we should consider this point to be the same as the old one and update.
            for j in objects.keys():
                object = objects[j]
                detections.sort(key=lambda x: dist(object, x))
                for d in detections:
                    distance = dist(d, object)
                    if distance < tolerance \
                            and d not in seen \
                            and j not in updated \
                            and not any([(dist(ob, d) < distance) for ob in objects.values() if ob != object]):
                        objects[j] = d
                        updated.append(j)
                        seen.append(d)

            # If the point was not near any existing objects, we check if it is within the given radius
            # of one of the entry points. If it is, we count it as a new object (a person who came
            # into the room). If not, it might be noise, but we count it as a potential object first
            # just to be safe.

            # Entry protocol update: Add entries to the potential object list as well. If we simply add detections
            # near the door to the object list, we run the risk of actualizing a false positive. The entries must
            # satisfy the same criteria as the other potential objects before they can be counted as real objects.
            for d in detections:
                for ex in exit_coordinates:
                    if euclidean_distance(d, ex) < exits_tolerance and d not in seen:
                        potential_objects.append(d)
                        seen.append(d)
                        break
                if d not in seen:
                    # print("Potential object registered. May be noise.")
                    potential_objects.append(d)

            mqtt_out_dict = {"detectionsWorld": [list(v) for v in objects.values()],
                             "detectionIDs": [str(k) for k in objects.keys()],
                             "lostIDs": lost_ids,
                             "deviceName": det_sensor,
                             'timestamp': timestamp,
                             "utcSecs": det_utc,
                             "utcUsecs": det_usec,
                             "detectionsLocal": det_local,
                             "detectionsWorld_RAW": [list(cd) for cd in detections]
                             }

        # Check if there is anyone in the object list. If there is, we look if the last place we had located them was
        # near an entry/exit point. If so, remove them from the list. If not, we may have ust lost the person
        # erroneously, so we do not update the object list.
        length = len(objects)
        objects = {k: v for k, v in objects.items()
                   if all([euclidean_distance(v, x) > exits_tolerance * exit_tol_scale
                           for x in exit_coordinates])}
        # if length > len(objects): print("Exit Registered")

        # Send to MQTT or write to a text file
        if (objects or (time.time() - last_send_time) % 10 == 0) and len(mqtt_out_dict) > 0:
            objects = {k: tuple(v) for k, v in objects.items()}
            objects = collapse(objects)
            time_delta = last_detection_time - mqtt_out_dict['timestamp']
            if abs(time_delta) >= 1000 and not detection_queue:
                mqtt_out_dict['timestamp'] = last_detection_time + time_delta
            payload = str(mqtt_out_dict)
            payload = re.sub("'", "\"", payload)
            if publish_to_mqtt:
                client.publish(mqtt_objects_out, payload=payload, qos=0)
            else:
                assert txt_file_path_out is not None
                with open(txt_file_path_out, "a+") as f:
                    f.write(payload)
                    f.write("\n")
            last_send_time = time.time()

        time.sleep(0.001)


def reset():
    global objects
    global potential_objects
    global object_id
    global mqtt_out_dict
    global detection_queue

    objects = {}
    mqtt_out_dict = {}
    potential_objects = []
    object_id = 0
    detection_queue = deque([])


def collapse(input_dict):
    global object_id
    vals = list(input_dict.values())
    unique_vals = set(vals)
    seen = {}
    kill = []
    for v in unique_vals:
        seen[v] = False
    for k, v in input_dict.items():
        if seen[v]:
            kill.append(k)
        else:
            seen[v] = True
    out = {k: v for k, v in input_dict.items() if k not in kill}
    if len(out) > 0: object_id = max(list(out.keys())) + 1
    return out


def mqtt_processes(address, topic_detect_in, usn, pw, client):
    global data_queue
    global detection_queue
    global post_queue
    global detection_list
    global device
    detection_list = []
    device = ""

    def on_subscribe(client, userdata, mid, granted_qos):
        print("Subscribed: " + str(mid) + " " + str(granted_qos))

    def on_message(client, userdata, msg):
        global detection_list
        global device
        try:
            message = json.loads(msg.payload)
            if "deviceName" in list(message.keys()):
                if not device: device = message['deviceName']
                if len(message['detectionsWorld']) > 0:
                    detection_list.append(message)
                if len(detection_list) >= 4:
                    detection_queue.append(detection_list)
                    detection_list = []

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
    client.subscribe(topic_detect_in, qos=1)
    # client.subscribe(mqtt_raw_in, qos=1)
    client.loop_forever()


def euclidean_distance(pt1, pt2):
    return np.sqrt(((pt2[0] - pt1[0]) ** 2) + ((pt2[1] - pt1[1]) ** 2))


def intersect(l0, l1):
    theta = np.radians(4)
    s = np.sin(theta)
    c = np.cos(theta)
    rotation = [[c, -s], [s, c]]
    rotation = np.asarray(rotation).reshape((2, 2))

    vec00 = np.matmul(np.array(l0[0]).T, rotation)
    vec01 = np.matmul(np.array(l0[1]).T, rotation)
    vec10 = np.matmul(np.array(l1[0]).T, rotation)
    vec11 = np.matmul(np.array(l1[1]).T, rotation)
    x0, y0 = vec00
    x1, y1 = vec01
    x2, y2 = vec10
    x3, y3 = vec11

    m0 = (y1 - y0) / (x1 - x0)
    b0 = y0 - (m0 * x0)

    m1 = (y3 - y2) / (x3 - x2)
    b1 = y2 - (m1 * x2)

    if np.around(m1, 3) == np.around(m0, 3):
        return False

    x_intersect = (b1 - b0) / (m0 - m1)
    xs = [x0, x1]
    xs.sort()
    xs1 = [x2, x3]
    xs1.sort()
    if np.around((m0 * x_intersect) + b0, 2) == np.around((m1 * x_intersect) + b1, 2) and \
            (xs[0] < x_intersect < xs[1]) and (xs1[0] < x_intersect < xs1[1]):
        return True
    return False


def make_graph(config_json, output_path):
    with open(config_json, "r") as f:
        config = json.load(f)
    walls = config['coordinates'][0]['walls']

    room_dims = config['room dimensions']

    space = 0.25
    diag = np.sqrt(2 * (space ** 2))

    n = int(room_dims[0] * (1 / space))
    m = int(room_dims[1] * (1 / space))
    positions = {}
    file_out = {}

    for i in range(m):
        for j in range(n):
            name = (i * m) + j
            positions[name] = [space * j, room_dims[1] - (space * i)]
    for i in range(m):
        for j in range(n):
            name = (i * m) + j
            neighbors = []
            diag_neighbors = []

            if i > 0:
                neighbor_up = ((i - 1) * m) + j
                if not any([intersect([positions[name], positions[neighbor_up]], wall) for wall in walls]):
                    neighbors.append(neighbor_up)
                if j < (n - 1):
                    neighbor_diag_right = ((i - 1) * m) + j + 1
                    if not any([intersect([positions[name], positions[neighbor_diag_right]], wall) for wall in walls]):
                        diag_neighbors.append(neighbor_diag_right)
                if j > 0:
                    neighbor_diag_left = ((i - 1) * m) + j - 1
                    if not any([intersect([positions[name], positions[neighbor_diag_left]], wall) for wall in walls]):
                        diag_neighbors.append(neighbor_diag_left)

            if j > 0:
                neighbor_left = (i * m) + j - 1
                if not any([intersect([positions[name], positions[neighbor_left]], wall) for wall in walls]):
                    neighbors.append(neighbor_left)

            if j < (n - 1):
                neighbor_right = (i * m) + j + 1
                if not any([intersect([positions[name], positions[neighbor_right]], wall) for wall in walls]):
                    neighbors.append(neighbor_right)

            if i < (m - 1):
                neighbor_down = ((i + 1) * m) + j
                if not any([intersect([positions[name], positions[neighbor_down]], wall) for wall in walls]):
                    neighbors.append(neighbor_down)
                if j < (n - 1):
                    neighbor_diag_right = ((i + 1) * m) + j + 1
                    if not any([intersect([positions[name], positions[neighbor_diag_right]], wall) for wall in walls]):
                        diag_neighbors.append(neighbor_diag_right)
                if j > 0:
                    neighbor_diag_left = ((i + 1) * m) + j - 1
                    if not any([intersect([positions[name], positions[neighbor_diag_left]], wall) for wall in walls]):
                        diag_neighbors.append(neighbor_diag_left)

            params = {"position": positions[name], "neighbors": [(n, space) for n in neighbors]}
            params['neighbors'].extend([(n, diag) for n in diag_neighbors])
            file_out[name] = [params]

    with open(output_path, "w") as f:
        json.dump(file_out, f, indent=4)


def check_edges(local, world, device_ids):
    edge_detections = []
    edge_detection_device_ids = []
    duplicates_full = []
    replacements = []
    for i, d in enumerate(local):
        if any([(0 < d[p] <= 0.125 or 0.875 <= d[p] < 1) for p in range(2)]):
            edge_detections.append(i)
            edge_detection_device_ids.append(device_ids[i])
    for i in edge_detections:
        duplicates = []
        w0 = world[i]
        device_id = device_ids[i]
        duplicates.extend([x for x in edge_detections if x != i
                           and device_id != device_ids[x]
                           and euclidean_distance(w0, world[x]) < 0.5
                           and x not in duplicates])
        if len(duplicates) > 0: duplicates.append(i)
        temp = [tuple(world[d]) for d in duplicates]
        if len(temp) > 0:
            # print("Collapsing duplicates (378)")
            new_coords = list(np.sum(temp, axis=0) / 2)
            replacements.append(new_coords)
            duplicates_full.extend(duplicates)

    for r in replacements: world.append(r)
    return [tuple(w) for w in world]


def main():
    global data_queue
    global detection_queue
    global minimum_to_register
    global exit_tol_scale

    client = paho.Client()
    data_queue = deque([])
    detection_queue = deque([])

    parser = argparse.ArgumentParser()
    parser.add_argument('-config', default="cfg/sensor_spatial_config_idpSante.json")
    parser.add_argument('-tol', default="3")

    parser.add_argument("-ptmq", default="t")
    parser.add_argument('-mqad', default="ec2-34-222-201-1.us-west-2.compute.amazonaws.com")
    parser.add_argument('-mqus', default="")  # butlr
    parser.add_argument('-mqpw', default="")  # 2019Ted/
    parser.add_argument('-mqdi', default="")
    parser.add_argument('-mqdo', default="")
    parser.add_argument('-gpo', default="")
    parser.add_argument('-ets', default="0.8")
    parser.add_argument('-min', default="0")
    parser.add_argument('-tpo', default="")

    args = parser.parse_args()

    config = args.config
    tolerance = float(args.tol)

    publish_to_mqtt = args.ptmq == "t"
    mqtt_address = args.mqad
    mqtt_detections_in = args.mqdi
    mqtt_detections_out = args.mqdo
    mqtt_usn = args.mqus
    mqtt_pw = args.mqpw
    graph = args.gpo
    exit_tol_scale = float(args.ets)
    minimum_to_register = float(args.min)

    txt_path = args.tpo
    _thread.start_new_thread(mqtt_processes, (mqtt_address, mqtt_detections_in, mqtt_usn, mqtt_pw, client))
    live_processing(config, tolerance, mqtt_objects_out=mqtt_detections_out, client=client,
                    publish_to_mqtt=publish_to_mqtt, graph_path_out=graph, txt_file_path_out=txt_path)


if __name__ == '__main__':
    main()
