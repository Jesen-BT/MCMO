import numpy
import pandas as pd
import numpy as np

data = pd.read_csv("data/Elec.csv")
column = data.columns
data = data.values
Split_point = [[0.1, 0.4, 0.9], [0.3, 0.8, 0.9], [0.5, 0.6, 0.7], [0.1, 0.2, 0.5]]

Stream_sample = int(len(data)/4)

target = []
s1 = []
s2 = []
s3 = []
Streams = [target, s1, s2, s3]

data_batch =[data[:Stream_sample],
       data[Stream_sample:2*Stream_sample],
       data[2*Stream_sample:3*Stream_sample],
       data[3*Stream_sample:4*Stream_sample]]

for i in range(4):
    batch = data_batch[i]
    Point = Split_point[i]
    mean_data_batch = numpy.mean(batch, axis=0)
    dis = np.sum(np.square(batch - mean_data_batch), axis=1)
    div = np.var(dis)

    P = dis / (2 * div)
    index = np.argsort(P)
    sort_data = batch[index]
    Split = [batch[:int(Stream_sample*Point[0])],
             batch[int(Stream_sample*Point[0]):int(Stream_sample*Point[1])],
             batch[int(Stream_sample*Point[1]):int(Stream_sample*Point[2])],
             batch[int(Stream_sample*Point[2]):]]

    for j in range(4):
        if i == 0:
            Streams[j] = Split[j]
        else:
            Streams[j] = np.append(Streams[j], Split[j], axis=0)


name = ["Target", "Source1", "Source2", "Source3"]
for i in range(len(name)):
    Streams[i] = pd.DataFrame(Streams[i])
    Streams[i].columns = column
    Streams[i].to_csv("data/Elec/"+name[i]+".csv")