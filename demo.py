from MCMO import MCMO
from skmultiflow.data.file_stream import FileStream
from evaluator import ClassificationMeasurements
import numpy as np
import matplotlib.pyplot as plt

model = MCMO(source_number=3, initial_beach=200, max_pool=5)

Tstream = FileStream("data/Weather/Target.csv")
S1stream = FileStream("data/Weather/Source1.csv")
S2stream = FileStream("data/Weather/Source2.csv")
S3stream = FileStream("data/Weather/Source3.csv")

count = 0
data_size = 0
result_list = []
result_list2 = []
t_list = []
data_window = [[], []]

index = -1

matrix = ClassificationMeasurements(dtype=np.float64)
souce = [S1stream, S2stream, S3stream]
while Tstream.has_more_samples():
    index = 0
    souce_index = 0
    for i in range(3):
        X, y = souce[i].next_sample()
        model.source_fit(X=X, y=y, order=i)
    X, y = Tstream.next_sample()
    pre = model.predict(X=X)


    matrix.add_result(y, int(pre))

    model.partial_fit(X=X, y=y)
    data_size = data_size + 1

    if data_size % 100 == 0.:
        result_list.append(matrix.get_accuracy())
        matrix = ClassificationMeasurements(dtype=np.float64)
        t_list.append(data_size)
        plt.plot(t_list, result_list, c='r', ls='-', marker='o', mec='b', mfc='w', label='MCMO')
        if index == -1:
            plt.legend()
            index = 0
        plt.pause(0.01)
        print(data_size)

plt.show()

print(np.mean(result_list))
