import copy as cp
import geatpy as ea
from skmultiflow.core import BaseSKMObject, ClassifierMixin, MetaEstimatorMixin
from skmultiflow.trees import HoeffdingAdaptiveTreeClassifier
from skmultiflow.data.file_stream import FileStream
from skmultiflow.drift_detection import DDM
from OptAlgorithm import MyProblem, Feature_Reduce
from GMM import DGMM
import numpy as np

class MCMO(BaseSKMObject, ClassifierMixin, MetaEstimatorMixin):

    def __init__(self, base_classifier=HoeffdingAdaptiveTreeClassifier(), detector=DDM(min_num_instances=100), source_number=1, initial_beach=100, max_pool=5):
        self.base_classifier = base_classifier
        self.source_number = source_number
        self.initial_beach = initial_beach
        self.max_pool = max_pool
        self.detector = detector

        self.source_classifiers = []
        self.classifier_pool = []
        self.drift_detectors = []
        self.probability_set = []

        self.S_list = []
        self.LS_list = []

        for i in range(self.source_number):
            self.S_list.append([])
            self.LS_list.append([])

        self.i = -1
        self.op = 0

    def partial_fit(self, X, y, classes=None, sample_weight=None):
        if self.i == -1:
            N, D = X.shape
            self.D = D
            self.T = np.zeros((self.initial_beach, self.D))

            self.i = 0

        if self.i < self.initial_beach:
            for i in range(len(X)):
                self.T[self.i] = X[i]
                self.i = self.i + 1
        else:
            Tx = Feature_Reduce(X, self.solution)
            probability = self.gmm.evaluation_weight(Tx)
            self.i = self.i + len(probability)
            for i in range(len(probability)):
                self.probability_set.pop(0)
                self.probability_set.append(probability[i])
                mean = np.mean(self.probability_set)
                if mean < self.drift_threshold:
                    self.op = 0
                    self.i = 0
                    self.S_list = []
                    self.LS_list = []
                    for j in range(self.source_number):
                        self.S_list.append([])
                        self.LS_list.append([])

        if self.i == self.initial_beach:
            self.Model_initialization()
            self.op = 1

    def source_fit(self, X, y, order):
        if self.op == 0:
            for i in range(len(X)):
                self.S_list[order].append(X[i])
                self.LS_list[order].append(y[i])

        else:
            Tx = Feature_Reduce(X, self.solution)
            weight = self.gmm.evaluation_weight(Tx)
            pre = self.source_classifiers[order].predict(Tx)
            for i in range(len(pre)):
                if pre[i] == y[i]:
                    self.drift_detectors[order].add_element(0)
                else:
                    self.drift_detectors[order].add_element(1)
                if self.drift_detectors[order].detected_change():
                    self.classifier_pool.append(self.source_classifiers[order])
                    self.drift_detectors[order] = cp.deepcopy(self.detector)
                    self.source_classifiers[order] = cp.deepcopy(self.base_classifier)
                    if len(self.classifier_pool) > self.max_pool:
                        self.classifier_pool.pop(0)

            self.source_classifiers[order].partial_fit(X=Tx, y=y)


    def Feature_Reduce(self, X, individual):
        f = np.sum(individual)
        Trans = np.zeros((self.D, f))

        c = 0
        for i in range(len(individual)):
            if individual[i] == 1:
                Trans[i][c] = 1
                c = c + 1

        Tx = np.dot(X, Trans)
        return Tx

    def Model_initialization(self):
        problem = MyProblem(n=self.D, T=self.T, S_list=self.S_list, LS_list=self.LS_list, d=self.D)
        algorithm = ea.moea_NSGA2_templet(problem,
                                          ea.Population(Encoding='BG', NIND=50),
                                          MAXGEN=50,  # 最大进化代数
                                          logTras=0)  # 表示每隔多少代记录一次日志信息，0表示不记录。
        algorithm.mutOper.Pm = 0.2  # 修改变异算子的变异概率
        algorithm.recOper.XOVR = 0.9  # 修改交叉算子的交叉概率
        res = ea.optimize(algorithm, verbose=False, drawing=0, outputMsg=False, drawLog=False, saveFlag=False)[
            'Vars']
        self.solution = res[0]
        for var in res:
            if sum(var) > sum(self.solution):
                self.solution = var

        print(self.solution)

        Tx = Feature_Reduce(self.T, self.solution)
        self.gmm = DGMM(n_components=5, random_state=0).fit(Tx)

        initial_confidence = self.gmm.evaluation_weight(Tx)
        self.drift_threshold = np.mean(initial_confidence) - 3*(np.std(initial_confidence))
        self.probability_set = list(initial_confidence)

        self.source_classifiers = []
        self.classifier_pool = []
        self.drift_detectors = []
        for i in range(self.source_number):
            Sx = Feature_Reduce(np.array(self.S_list[i]), self.solution)
            weight = self.gmm.evaluation_weight(Sx)
            base_classifier = cp.deepcopy(self.base_classifier)
            base_classifier.partial_fit(X=Sx, y=np.array(self.LS_list[i]))
            self.source_classifiers.append(base_classifier)
            self.drift_detectors.append(cp.deepcopy(self.detector))

    def predict_proba(self, X):
        N, D = X.shape
        votes = np.zeros(N)

        if len(self.source_classifiers) <= 0:
            return votes

        Tx = Feature_Reduce(X, self.solution)

        if len(self.source_classifiers) == 0:
            for h_i in self.source_classifiers:
                votes = votes + 1. / len(self.source_classifiers) * h_i.predict(Tx)
        else:
            for h_i in self.source_classifiers:
                votes = votes + 1. / (len(self.source_classifiers) + len(self.classifier_pool)) * h_i.predict(Tx)
            for h_i in self.classifier_pool:
                votes = votes + 1. / (len(self.source_classifiers) + len(self.classifier_pool)) * h_i.predict(Tx)

        return votes

    def predict(self, X):
        votes = self.predict_proba(X)
        return (votes >= 0.5) * 1.




if __name__ == '__main__':
    model = MCMO(source_number=3, initial_beach=200)
    stream = FileStream("DATA/weather.csv")
    n_samples = 0
    while n_samples < 800 and stream.has_more_samples():
        n_samples = n_samples + 1
        X, y = stream.next_sample()
        for i in range(3):
            model.source_fit(X=X, y=y, order=i)
        model.partial_fit(X=X, y=y)

    X, y = stream.next_sample(10)
    print(model.predict(X))
    print(y)









