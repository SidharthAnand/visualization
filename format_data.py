import argparse
import numpy as np


def unify(data_path, label_path=None, num_in_frame=2, left=0, right=0):
    if num_in_frame == 1:
        if "lying_" in data_path:
            left = right = 2
        elif "standing_" in data_path:
            left = right = 1
        else:
            left = right = 0
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
            try:
                label_line = eval(labels[label_start])
                label_time = label_line["timestamp"]
                label_mac = label_line["ID"]
                boxes = []
                formatted_boxes = []
                while label_time <= timestamp:
                    if label_time == timestamp and label_mac == mac:
                        boxes = label_line['bounding box']
                        break
                    else:
                        label_start += 1
                        try:
                            label_time = eval(labels[label_start])["timestamp"]
                        except IndexError:
                            label_time = timestamp
                for box in boxes:
                    # print(box)
                    x_center = (box[1][0] + box[0][0]) * 0.5
                    y_center = (box[1][1] + box[0][1]) * 0.5
                    width = np.abs(box[1][0] - box[0][0])
                    height = np.abs(box[1][1] - box[0][1])
                    formatted_boxes.append([x_center, y_center, width, height])
                if formatted_boxes:
                    formatted_boxes.sort(key=lambda x: x[0])
                line_out["bbox"] = formatted_boxes
                if num_in_frame == 1:
                    line_out["category_id"] = [left] * len(boxes)
                elif num_in_frame == 2:
                    line_out["category_id"] = [left, right]
                else:
                    raise ValueError
            except IndexError:
                pass
        text_lines_out.append(str(line_out) + "\n")
    with open(unified_path, "a+") as f:
        f.writelines(text_lines_out)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-d")
    parser.add_argument("-l")
    parser.add_argument("-p", default="2")
    parser.add_argument("-left", default="0")
    parser.add_argument("-right", default="0")
    args = parser.parse_args()
    in_frame = eval(args.p)
    left = eval(args.left)
    right = eval(args.right)
    d = args.d
    l = args.l
    unify(d, l, num_in_frame=in_frame, left=left, right=right)


if __name__ == "__main__":
    main()
