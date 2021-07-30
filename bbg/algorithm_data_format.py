

def transform32(input_file_path):
    output_file_path = input_file_path[:-4] + "_transformed.txt"
    f=open(input_file_path, 'r')
    res=f.read()

    f.close()
    res=res.replace('[b"({','{')
    res=res.replace(',)", b"(','\n')
    res=res.replace('},)"]','}')
    res=res.replace('array(','')
    res=res.replace(')','')
    res = res.replace("{'fields': ", "")
    res = res.replace(", 'name': 'notifData'", " ")
    res = res.replace(", 'macAddress': 'xxx'}", " ")
    res = res.replace("{'data':", '{"id": "xxx","p":')
    res = res.replace("'timestamp': ", '"t":')
    s=open(output_file_path,'a+')
    s=s.write(res)
    return output_file_path


def transform8(input_file_path):
    f = open(input_file_path, 'r')
    output_file_path = input_file_path[:-4] + "_transformed.txt"

    res = f.read()
    res = res.replace("[[b'{", "{")
    res = res.replace("}']]", "}")
    res = res.replace("', b'", '\n')
    res = res.replace("'], [b'", '\n')

    s = open(output_file_path, 'a+')
    s = s.write(res)
    return output_file_path