import numpy as np
import json
from json import JSONEncoder


class NumpyArrayEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return JSONEncoder.default(self, obj)


def process_fun(input_file):
    file_path_read = input_file
    file_path_write = input_file[:-4] + "_final.json"

    f = open(file_path_read, 'r')

    while True:
        txt_s = f.readline()
        if txt_s == '':
            break
        dict_line = eval(txt_s)
        try:
            data = dict_line["p"]
            macid = dict_line["id"]
            tstamp = dict_line["t"]
            datarev = np.array(data).reshape(1024)
            # print(datarev)
            wline = {"id": macid, "p": datarev * 4, "t": tstamp}
            # print(wline)
            jsObj = json.dumps(wline, cls=NumpyArrayEncoder)
            # fileObject = open('1024trans.json', 'w')
            fileObject = open(file_path_write, 'a+')
            fileObject.write(jsObj + '\n')
            # fileObject.close()
        except:
            print('eee')

    fileObject.close()
    return file_path_write
