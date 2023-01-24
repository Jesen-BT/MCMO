from MCMO import MCMO
from skmultiflow.data.file_stream import FileStream
from sklearn.metrics import accuracy_score
import time

model = MCMO(source_number=3, initial_beach=200, max_pool=10)
Tstream = FileStream("DATA/Elec/Target.csv")
S1stream = FileStream("DATA/Elec/Source1.csv")
S2stream = FileStream("DATA/Elec/Source2.csv")
S3stream = FileStream("DATA/Elec/Source3.csv")


souce = [S1stream, S2stream, S3stream]
begin = time.time()
n_samples = 0
while n_samples < 1000 and Tstream.has_more_samples():
    n_samples = n_samples + 1
    for i in range(3):
        X, y = souce[i].next_sample()
        model.source_fit(X=X, y=y, order=i)
    X, y = Tstream.next_sample()
    model.partial_fit(X=X, y=y)
end = time.time()

X, y = Tstream.next_sample(100)
print("Acc=" + str(accuracy_score(y_pred=model.predict(X), y_true=y)))
print("Time=" + str(end-begin))
