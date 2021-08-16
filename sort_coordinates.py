import os


def sortByX(file):
    with open(file, "r+") as f:
        text = f.readlines()
    linesOut = []
    for line in text:
        d = eval(line)
        boxes = d["bbox"]
        boxes.sort(key=lambda x: x[0])
        d["bbox"] = boxes
        linesOut.append(str(d))
    outFile = file[:-4] + "_sorted.txt"
    with open(outFile, "w") as f:
        f.writelines(linesOut)


def main():
    unsorted = os.listdir("rsfd")
    for k in unsorted:
        print("Now sorting:", k)
        sortByX("rsfd/"+k)

if __name__ == "__main__":
    main()