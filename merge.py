import pandas as pd
from scipy.io import arff
import os

directory = "data/"

samples = []
for i in os.listdir(directory):
    if i[-4:] == "arff":
        arf = arff.loadarff(directory + i)
        to_add = pd.DataFrame(arf[0])
        to_add["year"] = int(i[0])
        to_add["class"] = [i.decode('utf-8') for i in to_add["class"]]
        samples.append(to_add)

data = pd.concat(samples, ignore_index=True)
data.to_csv("data/data.csv", index=False)
