from visualizer.visualizer import *
import time
import _thread
import paho.mqtt.client as paho


client = paho.Client()


def mqtt_processes(address, topic_raw_in, topic_detect_in, usn, pw):
    global data_queue
    global detection_queue
    global client
    global bufferMQ
    global sensor_timer
    global det_buffer
    global timestamp

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
            if type(message) is tuple: message = message[0]
            # print(json.dumps(message))
            if list(message.keys())[0] == 'fields':
                if message['name'] == 'notifData':
                    try: timestamp = message["fields"]["timestamp"]
                    except: timestamp = 0
                    try:
                        message["fields"]["data"] = (message["fields"]["data"]).tolist()
                    except:
                        pass
                    data_queue.append(str(message))

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
    client.loop_forever()


def collect_data(mqtt_address, data_topic, data_path_out, period=10):
    global data_queue

    big_data = False
    if "htps" in data_topic: big_data = True

    data_queue = deque([])
    start_time = time.time()
    i = start_time
    f = open(data_path_out, "a+")
    print(f"Beginning data collection from MQTT ({data_topic})")
    _thread.start_new_thread(mqtt_processes, (mqtt_address, data_topic, None, "butlr", "2019Ted/"))
    lines = []
    while True:
        while len(data_queue) > 0:
            packet = data_queue.popleft()
            lines.append(packet)
            f.write(packet + "\n")
        i = time.time()
        if i - start_time % 30 == 0: print(f"{np.around((i-start_time) / 60., 3)} of "
                                           f"{np.around(period / 60., 3)} minutes elapsed.")
        if i > start_time + period: break
    print(f"Collected {len(lines)} lines of data in {period} seconds.\n\n")
    # print("Automatically generating bounding boxes on the collected data...\n")
    cmd32 = f'python bbg/Butlr_PoC_one.py -m synthetic_data_one -dp {data_path_out} -n t -amm t -mmdl 0.1 ' \
            f'-imin 0 -imax 10 -wamm 1000 -famm 5 -abg 1 -rabg 9999999 -fabg 10 -sres "32*32" -eres "8*8" ' \
            f'-sc 0.1 -thh auto -thc -999 -pub f -be f -dr 7,10 -csh -0.4 -bsh -1.0 -cr2 0.03 -br2 0.08 ' \
            f'-dth 20,30 -ps 100 -bbp {data_path_out[:-4] + "boundingbox.txt"} -viz f'
    cmd8 = f'python bbg/Butlr_PoC_one_new88.py -m saved_data -dp {data_path_out} -mqid xxx  ' \
           f'-pub f -n t -amm t -mmdl 0.1 -imin 0 -imax 10 -wamm 1000 -famm 5 -abg 1 -rabg 9999999 -fabg 10 ' \
           f'-lt 100 -vt 10,50 -dmh both -dmc th -dr 2.5,3 -thh auto -thc -999 -thstdc 7.5 -thstdh 3 ' \
           f'-ds 0.2001,0.0001 -de 0.2001,0.9999 -be f -trk f -art t -tshc t -cfo f -cmsg seamless -sens 1 ' \
           f'-wldcd f -dtcmq f -ps 1 -be f -viz t -bbp {data_path_out[:-4] + "boundingbox.txt"} -dth 10,20'
    # if big_data:
    #     os.system(cmd32)
    # else:
    #     os.system(cmd8)
    print("Success")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-mqba", default="ec2-54-245-187-200.us-west-2.compute.amazonaws.com")
    parser.add_argument("-mqdi", default="butlr/htps/test")
    parser.add_argument("-path", help="Output path for data text file.")
    parser.add_argument("-time", default="600", help="Time (in seconds) to collect data.")

    args = parser.parse_args()
    mqba = args.mqba
    mqdi = args.mqdi
    t = eval(args.time)
    path = args.path if args.path else None
    assert path is not None, "Please specify a path to which the data should be written."
    collect_data(mqba, mqdi, path, period=t)

if __name__ == "__main__": main()
