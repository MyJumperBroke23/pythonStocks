import csv
import numpy as np


def readCSV(path: str, cols: list = [1, 2, 3, 4, 5]):
    arr = []
    with open(path) as file:
        reader = csv.reader(file, delimiter=',')
        linecount = 0
        for row in reader:
            if linecount > 1:
                j = [float(row[i]) for i in cols]
                arr.append(j)
            linecount += 1
    return np.array(arr)

def genData(data):
    arrayLen = len(data)*(len(data)-1) / 2
    returnArray = []
    for firstPoint in range(len(data)):
        for numPoints in range(1, len(data) - firstPoint):
            input = data[firstPoint:firstPoint + numPoints]
            label = data[firstPoint + numPoints]
            returnArray.append((input, label))
    assert len(returnArray) == arrayLen
    return returnArray

