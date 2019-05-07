from dataio import readCSV
import numpy as np
j = readCSV("data/AAPL3.csv")
j = np.flip(j,0)
print(j[0])