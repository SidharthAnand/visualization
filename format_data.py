import argparse
import numpy as np


def unify(data_path, label_path=None):
    if "lying_" in data_path:
        pose = 2
    elif "standing_" in data_path:
        pose = 1
    else:
        pose = 0
    data = open(data_path, "r").readlines()
    if len(data) == 1:
        try:
            data = eval(data[0])
        except:
            print(len(data))
            print("something went wrong")
    if type(data) is list and all([type(x) is list for x in data]):
        full_data = []
        for j in data:
            full_data += j
        data = full_data
    if label_path:
        labels = open(label_path, "r").readlines()
    unified_path = data_path[:-4] + "_unified.txt"
    label_start = 0
    text_lines_out = []
    for line in data:
        line_out = {
            "image": [],
            "bbox": [],
            "category_id": [],
            "timestamp": 0,
            "mac_address": "",
            "normalized": "false"
        }
        array = lambda x: np.asarray(x)
        d = eval(line)
        if type(d) is tuple: d = d[0]
        if d["name"] != "notifData": continue
        timestamp = d['timestamp']
        mac = d["fields"]["macAddress"]

        line_out["timestamp"] = timestamp
        line_out["mac_address"] = mac
        try:
            line_out["image"] = d["fields"]["data"].tolist()
        except:
            line_out["image"] = d["fields"]["data"]
            if len(line_out["image"]) == 72:
                line_out["image"] = line_out["image"][6:-2]
            elif len(line_out["image"]) == 74:
                line_out["image"] = line_out["image"][6:-4]
            assert len(line_out["image"]) in [64, 32], len(line_out["image"])
            if len(line_out["image"]) == 64:
                line_out["image"] = [[line_out["image"][8*k + i] for i in range(8)] for k in range(8)]

        if label_path:
            label_line = eval(labels[label_start])
            label_time = label_line["timestamp"]
            label_mac = label_line["ID"]
            boxes = []
            formatted_boxes = []
            while label_time <= timestamp:
                if label_time == timestamp and label_mac == mac:
                    boxes = label_line['bounding box']
                    break
                label_start += 1
                try:
                    label_time = eval(labels[label_start])["timestamp"]
                except IndexError:
                    label_time = timestamp
            for box in boxes:
                x_center = (box[1][0] + box[0][0]) * 0.5
                y_center = (box[1][1] + box[0][1]) * 0.5
                width = np.abs(box[1][0] - box[0][0])
                height = np.abs(box[1][1] - box[0][1])
                formatted_boxes.append([x_center, y_center, width, height])
            line_out["bbox"] = formatted_boxes
            line_out["category_id"] = [pose] * len(boxes)
        text_lines_out.append(str(line_out) + "\n")
    with open(unified_path, "a+") as f:
        f.writelines(text_lines_out)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-d")
    parser.add_argument("-l")
    args = parser.parse_args()
    d = args.d
    l = args.l
    unify(d, l)


if __name__ == "__main__":
    main()
