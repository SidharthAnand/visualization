import time
import os
# from networkx.algorithms.operators.binary import intersection
import paho.mqtt.client as paho
import argparse
import urllib
import json
import datetime
import calendar
import requests
import ast
import dateutil.parser
import pprint
import operator
import numpy
import re
import multiprocessing
# import networkx
# from networkx.algorithms.components.connected import connected_components
from calendar import timegm
from collections import defaultdict


def format_headcount_raw(data):
    answer = {}
    curr_point = ((data[0][6], data[0][2]))
    for datapoint in data[1:]:
        if (datapoint[6], datapoint[2]) in answer:
            answer[(datapoint[6], datapoint[2])].append((datapoint[4], datapoint[3]))
        else:
            answer[(datapoint[6], datapoint[2])] = [(datapoint[4], datapoint[3])]
    # pprint.pprint(answer)

    final_answer = []
    for i in answer.keys():
        ins = int([t[1] for t in answer[i] if t[0] == 'in'][0])
        outs = int([t[1] for t in answer[i] if t[0] == 'out'][0])
        mqtt_message = {
            "device_id": i[0],
            "date_time": i[1].replace("T", " ")[:-5],
            "timestamp": int(dateutil.parser.parse(i[1]).timestamp() * 1000),
            "trajectory": [eval(t[1])[0] for t in answer[i] if t[0] == 'trajectory'],
            "exit": [eval(t[1])[0] for t in answer[i] if t[0] == 'exit'],
            "in_out": [ins,outs]
        }


        final_answer.append(mqtt_message)
    final_answer.sort(key = lambda i: i['timestamp'])
    return final_answer




def make_url(field='all', api_key='de300c74acc6d2572288c64b42e265d0', newdb = True):
    # theurl = f'https://mybutlr.io/api/{datatype}_historical_data?database={database}&field={field}&deviceID={deviceID}&time_range={time_range[0]}_{time_range[1]}&api_key={api_key}&replay=yes&old=False'

    # if locationID != "":
    #     theurl = f'https://mybutlr.io/api/{datatype}_historical_data?database={database}&field={field}&deviceID={deviceID}&time_range={time_range[0]}_{time_range[1]}&api_key={api_key}&replay=yes&old=False&locationID={locationID}'

    # if newdb:
    #https://y02xuncz98.execute-api.us-west-2.amazonaws.com
    #https://ln99u6l3q5.execute-api.us-west-2.amazonaws.com
    theurl = f'https://y02xuncz98.execute-api.us-west-2.amazonaws.com/prod/api/v1/streams?data_type={datatype}&database={database}&time_range={time_range[0]}_{time_range[1]}&device_id={device_id}&limit=100&skip=0&pod_id={pod_id}&data_format=json&key={api_key}&restream=True'

    print(theurl)

    return theurl

def get_data_from_url(theurl):
    response = requests.get(theurl)
    data = response.text

    return data

def to_unix_time(timestamp):

    timestamp = str(dateutil.parser.parse(timestamp))

    try:
        epoch_time = calendar.timegm(time.strptime(timestamp[:-6], "%Y-%m-%d %H:%M:%S.%f"))
        epoch_time = (epoch_time*1000000) + int(timestamp[20:26])
    except:
        epoch_time = calendar.timegm(time.strptime(timestamp[:-6], "%Y-%m-%d %H:%M:%S"))
        epoch_time = (epoch_time*1000000)
    return epoch_time



def make_payloads():
    #make the url and get the data from it
    theurl = make_url()
    data_from_url = get_data_from_url(theurl)

    global error
    error = "Success!"
    #The code below was calling get_data_from_url twice so replaced it with data_from_url as it's supposed to be.
    #This code block is to check the response from the API to see if it is a valid data response.
    try:
        response = ast.literal_eval(data_from_url)
    except:
        if data_from_url == "No Data" or data_from_url == '':
            error = "There is no data as a result of this query"
            return
        else:
            pass

    try:
        data = response['data']
        print("Got the data from the database via API call, please wait for processing. May take a few minutes!")
        if data == 'No Data' or data == '':
            print('No data as a result of this query')
            return
    except Exception as e:
        # print(response)
        error = "Probably and API error, perhaps Internal Server Error 500, please contact Arihan for this!"
        print(e)
        return


    if datatype == 'headcount_raw':
        if data[-1] == []:
            data.pop()
        headcount_txt = format_headcount_raw(data)
        b = int(dateutil.parser.parse(time_range[0]).timestamp() * 1000)
        e = int(dateutil.parser.parse(time_range[1]).timestamp() * 1000)
        begin_writing(None, headcount_txt, b, e, datatype)
        return

    #We are gathering the timestamps and matching data points from the data, with intentions to send them with proper pause times in between data points.
    timestamps = [(to_unix_time(i['_time']), i) for i in data if (len(i) > 0) and (i['_time'] not in ['dateTime:RFC3339','false','true', '', '_time'])]
    sending_array = []

    curr_time_index = 0
    timestamps.sort(key = operator.itemgetter(0))

    first_time = timestamps[0][0]
    last_time = timestamps[-1][0]

    #Building a initial sending array for all data points.
    curr_time = timestamps[0][0]
    sending_array.append([timestamps[0]])
    for i in range(1, len(timestamps)):

        if timestamps[i][0] == curr_time:

            sending_array[curr_time_index].append(timestamps[i])
        else:
            curr_time_index += 1

            sending_array.append([timestamps[i]])
            curr_time = timestamps[i][0]



    final_array = []
    final_times = []
    #Turn format of individual sensor and result data points from API response back into MQTT format.
    for arr in sending_array:

        if datatype == 'detection_raw':

            payload = {
                    "timestamp": int(arr[0][0]/1000),
                    "deviceName": arr[0][1]["device_id"]
                }

            for k in range(len(arr)):
                payload[str(arr[k][1]['_field'])] = ast.literal_eval(arr[k][1]["_value"])

            final_times.append(arr[0][0])

            final_array.append(payload)

        else:
            for k in range(len(arr)):
                payload = {
                    "fields": {
                        "macAddress": arr[k][1]["device_id"],
                        "srcPort": 61626,
                        "utcUsecs": int(arr[k][0]/1000),
                        "utcSecs": int(arr[k][0]/1000000),
                        "dstPort": 61626
                        },
                    "manager": "/dev/ttyUSB3",
                    "name": 'notifData',
                    "timestamp": int(arr[k][0]/1000),
                }
                if arr[k][1]["_field"] == 'therm_data':
                    data = [int(i * 4) for i in ast.literal_eval(arr[k][1]["_value"])]
                    for i in range(6):
                        data.insert(0,0)
                    for i in range(2):
                        data.append(0)
                    payload['fields']['data'] = data
                    final_times.append(arr[k][0])
                    final_array.append(payload)


    pause_times = []

    #new method to calculate the difference between the timestamp at position i to the first timestamp
    for k in range(len(final_times)):
        pause_times.append((final_times[k] - final_times[0])/1000000.0)


    #If we want to stream both raw and results, we want to return both arrays and synchronize the begin stream methods so they stream together.

    if both:
        return (pause_times, final_array, last_time, first_time, datatype)

    #Begin stream.

    if save_to_text == 'True':
        begin_writing(pause_times, final_array, last_time, first_time, datatype)
    else:
        begin_stream(pause_times, final_array, last_time, first_time, datatype)



def begin_stream(wait_times, sending_array, last_time, first_time, datatype):
    global client
    msg_count = 0
    error_count = 0
    client.loop_start()
    time_of_stream = (last_time - first_time)/1000000.0
    wait_times[:] = [wt / speed for wt in wait_times]

    mytopic = topic + '/' + datatype
    if type_to_wait == datatype:
        time.sleep(time_for_sync / speed)
    cnt = 0
    print(f'Now beginning {datatype} stream')
    begin = time.time()
    while True:
        if (wait_times[0] <= (time.time() - begin)):
            cnt += 1
            (rc, mid) = client.publish(mytopic, payload=json.dumps(sending_array.pop(0)).encode('utf-8'), qos=0)
            if rc == 0:
                msg_count += 1
            else:
                error_count +=1
            wait_times.pop(0)
        if len(sending_array) == 0:
            print(f"Time to send the {datatype} messages: ", time.time() - begin)
            print(f"Time difference in first and last message of {datatype}:", time_of_stream)
            print(f"Number of {datatype} messages streamed to MQTT: {msg_count}")
            print(f"Number of {datatype} messages failed to stream to MQTT: {error_count}")
            print(cnt)
            return

def begin_writing(wait_times, sending_array, last_time, first_time, datatype):

    returnfolder = f"{datatype}"

    if not os.path.exists(returnfolder):
        os.mkdir(returnfolder)

    querieddata_filename = f"{datatype}/{database}_{pod_id}_{first_time}_{last_time}.txt"

    f = open(querieddata_filename, "w+")
    print(f.name)
    for message in sending_array:
        message = str(message)
        f.write(message + '\n')

    if push_to_S3:
        begin_push(querieddata_filename, pod_id, first_time, last_time)


def begin_push(querieddata_filename, pod_id, first_time, last_time):


    start_time = epoch_sec_to_date_time_str(first_time)
    end_time = epoch_sec_to_date_time_str(last_time)

    cmd = f'aws s3 cp {querieddata_filename} s3://{data_library}/{database}/{datatype}/{pod_id}/{start_time}~{end_time}.txt'
    os.system(cmd)


def date_time_str_to_epoch_sec(date_time_str):
    utc_time = time.strptime(date_time_str, '%Y-%m-%d %H:%M:%S')
    epoch_time = timegm(utc_time)
    return epoch_time

def epoch_sec_to_date_time_str(epoch_time):
    #print(epoch_time)
    epoch_time = epoch_time // 1000000
    print(str(datetime.datetime.utcfromtimestamp(epoch_time)).replace(" ", "_"))
    return str(datetime.datetime.utcfromtimestamp(epoch_time)).replace(" ", "_")


def main():
    #ArgumentParser
    parser = argparse.ArgumentParser()
    parser.add_argument("-mqus", "-mqtt_username", help="username for mqtt broker", default="butlr")
    parser.add_argument("-mqpw", "-mqtt_password", help="password for mqtt broker", default="2019Ted/")
    parser.add_argument("-mqtp", "-mqtt_topic", help="topic to publish replay to", default="replay_demoqueue")
    parser.add_argument("-mqad", help="mqtt address", default='ec2-54-245-187-200.us-west-2.compute.amazonaws.com')
    parser.add_argument("-db", "-database", help="database to replay data from", default="idpsante")
    parser.add_argument("-dt", "-datatype", help="datatype of the data: sensor_raw or detection_raw", default="detection_raw")
    parser.add_argument("-ts", "-time_start", help="start time", default="2021-02-05 19:00:00")
    parser.add_argument("-te", "-time_end", help="end time", default="2021-02-05 20:00:00")
    parser.add_argument("-did", help="device_id", default="all")
    parser.add_argument("-ps", help="playback speed", default=1.0)
    parser.add_argument("-lp", help="boolean if looping once finished is desired.", default=False)
    parser.add_argument("-hid", help="hive_id", default="30_03")
    parser.add_argument("-stt", help="boolean to save to textfile", default=False)
    parser.add_argument("-pts3", help="boolean to push to S3 bucket", default=False)
    parser.add_argument("-dl", help="S3 data library", default='arihansdatalibrary')



    args = parser.parse_args()
    #Global Variables
    global topic
    global client
    global database
    global datatype
    global device_id
    global time_range
    global speed
    global both
    global username
    global password
    global address
    global time_for_sync
    global type_to_wait
    global error
    global pod_id
    global save_to_text
    global push_to_S3
    global data_library

    type_to_wait = "temp"
    topic = args.mqtp


    client = paho.Client()
    client.username_pw_set(args.mqus, args.mqpw)
    RC = client.connect(args.mqad, 1883, keepalive=60)


    username = args.mqus
    password = args.mqpw
    address = args.mqad


    database = args.db

    datatype = args.dt

    timestart = args.ts

    timeend = args.te

    time_range = [timestart, timeend]

    device_id = args.did

    pod_id = args.hid

    save_to_text = args.stt

    push_to_S3 = args.pts3
    if push_to_S3 == "True":
        save_to_text = "True"

    speed = float(args.ps)

    data_library = args.dl


    time_difference = date_time_str_to_epoch_sec(timeend) - date_time_str_to_epoch_sec(timestart)

    if time_difference > 18000:
        error = "Query Time too long. Please query 5 hours or less!"
        print(error)
        return

    if args.lp != True and args.lp != False:
        try:
            loop = ast.literal_eval(args.lp)
        except:
            error = "Invalid parameter for loop. Requires Boolean"
            print(error)
            return

    loop = False

    if datatype == 'both':
        both = True
        datatype = 'sensor_raw'
        sensor_args = make_payloads()
        datatype = 'detection_raw'
        results_args = make_payloads()
        #This block of code is to set the global variables for how much time to wait for the sync of the first messages and which data type is the one to have to wait.
        if results_args == None:
            results_args = []
            print('No detection_raw results')
        if sensor_args == None:
            sensor_args = []
            print('No sensor_raw results')

        if len(results_args) == 5 and len(sensor_args) == 5:
            time_for_sync = (max(sensor_args[3], results_args[3]) - min(sensor_args[3], results_args[3]))/1000000
            if sensor_args[3] > results_args[3]:
                type_to_wait = 'sensor_raw'
            else:
                type_to_wait = 'detection_raw'
        else:
            return


        args = []
        if len(sensor_args) == 5:
            args.append(sensor_args)
        if len(results_args) == 5:
            args.append(results_args)
        #Begin multithreading process to send both datatypes at the same time.
        pool = multiprocessing.Pool()

        if save_to_text == 'True':
            pool.starmap(begin_writing, args)
        else:
            pool.starmap(begin_stream, args)
        pool.close()
        pool.join()
        print('Finished all processes!')
    else:
        both = False
        make_payloads()
    print(error)


if __name__ == '__main__':
    main()