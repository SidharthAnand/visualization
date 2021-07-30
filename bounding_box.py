from bbg.algorithm_data_format import transform8, transform32
from bbg.trans_format_total import process_fun
import argparse
import os


def generate_bb(data_path, size=32):
    try:
        if size == 32:
            input_file = transform32(data_path)
            algorithm_input_file = process_fun(input_file)
            cmd32 = f'python bbg/Butlr_PoC_one_stop.py -m synthetic_data_one -dp {algorithm_input_file} -n t -amm t -mmdl 0.1 ' \
                    f'-imin 0 -imax 10 -wamm 1000 -famm 5 -abg 1 -rabg 9999999 -fabg 10 -sres "32*32" -eres "8*8" ' \
                    f'-sc 0.1 -thh auto -thc -999 -pub f -be f -dr 7,10 -csh -0.4 -bsh -1.0 -cr2 0.03 -br2 0.08 ' \
                    f'-dth 20,30 -ps 100 -bbp {data_path[:-4] + "_boundingbox.txt"} -viz f'
            os.system(cmd32)
        else:
            algorithm_input_file = transform8(data_path)
            input_file = algorithm_input_file
            cmd8 = f'python bbg/Butlr_PoC_one_new88.py -m saved_data -dp {algorithm_input_file} -mqid xxx  ' \
                   f'-pub f -n t -amm t -mmdl 0.1 -imin 0 -imax 10 -wamm 1000 -famm 5 -abg 1 -rabg 9999999 -fabg 10 ' \
                   f'-lt 100 -vt 10,50 -dmh both -dmc th -dr 2.5,3 -thh auto -thc -999 -thstdc 7.5 -thstdh 3 ' \
                   f'-ds 0.2001,0.0001 -de 0.2001,0.9999 -be f -trk f -art t -tshc t -cfo f -cmsg seamless -sens 1 ' \
                   f'-wldcd f -dtcmq f -ps 1 -be f -viz f -bbp {data_path[:-4] + "_boundingbox.txt"} -dth 10,20'
            os.system(cmd8)
    except KeyboardInterrupt:
        pass
    if os.path.exists(algorithm_input_file): os.remove(algorithm_input_file)
    if os.path.exists(input_file): os.remove(input_file)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", default="")
    parser.add_argument("-s", default="32")

    args = parser.parse_args()
    path = args.p
    size = eval(args.s)
    generate_bb(path, size)

if __name__ == "__main__": main()