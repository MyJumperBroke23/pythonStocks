import csv
import numpy as np


def readCSV(path: str, cols: list = [1, 3, 4, 5]):
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
