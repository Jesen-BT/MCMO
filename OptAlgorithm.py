import numpy as np
import geatpy as ea
from skmultiflow.data.file_stream import FileStream
from skmultiflow.utils import get_dimensions
from sklearn import metrics
import copy as cp

def F1(X, y):
    labels = list(set(y))
    xClasses = {}
    for label in labels:
        xClasses[label] = np.array([X[i] for i in range(len(X)) if y[i] == label])
    meanAll = np.mean(X, axis=0)
    meanClasses = {}
    for label in labels:
        meanClasses[label] = np.mean(xClasses[label], axis=0)  # 1*n
    St = np.dot((X - meanAll).T, X - meanAll)
    Sw = np.zeros((len(meanAll), len(meanAll)))
    for i in labels:
        Sw += np.dot((xClasses[i] - meanClasses[i]).T, (xClasses[i] - meanClasses[i]))
    Sb = St - Sw
    try:
        d = np.dot(np.linalg.inv(Sw), Sb)
    except:
        d = np.dot(np.linalg.pinv(Sw), Sb)
    x = 1 / (np.trace(d)+0.0000000001)
    return x

def Feature_Reduce(X, individual):
    n = get_dimensions(X)[1]
    f = np.sum(individual)
    Trans = np.zeros((n, f))

    c = 0
    for i in range(len(individual)):
        if individual[i] == 1:
            Trans[i][c] = 1
            c = c + 1

    Tx = np.dot(X, Trans)
    return Tx

def mmd_rbf(X, Y, gamma=1.0):
    XX = metrics.pairwise.rbf_kernel(X, X, gamma)
    YY = metrics.pairwise.rbf_kernel(Y, Y, gamma)
    XY = metrics.pairwise.rbf_kernel(X, Y, gamma)
    return XX.mean() + YY.mean() - 2 * XY.mean()

class MyProblem(ea.Problem):  # 继承Problem父类
    def __init__(self, M=2, n=0, T=[], S_list=[], LS_list=[], d=0):
        self.T = T
        self.S_list = cp.deepcopy(S_list)
        self.LS_list = cp.deepcopy(LS_list)
        self.n = n

        for i in range(len(self.S_list)):
            self.S_list[i] = np.array(self.S_list[i])
            self.LS_list[i] = np.array(self.LS_list[i])

        name = 'MyProblem'  # 初始化name（函数名称，可以随意设置）
        Dim = d  # 初始化Dim（决策变量维数）
        maxormins = [1] * M  # 初始化maxormins（目标最小最大化标记列表，1：最小化该目标；-1：最大化该目标）
        varTypes = [1] * Dim  # 初始化varTypes（决策变量的类型，0：实数；1：整数）
        lb = [0] * Dim  # 决策变量下界
        ub = [1] * Dim  # 决策变量上界
        lbin = [1] * Dim  # 决策变量下边界（0表示不包含该变量的下边界，1表示包含）
        ubin = [1] * Dim  # 决策变量上边界（0表示不包含该变量的上边界，1表示包含）
        # 调用父类构造方法完成实例化
        ea.Problem.__init__(self, name, M, maxormins, Dim, varTypes, lb, ub, lbin, ubin)

    def evalVars(self, Vars):  # 目标函数
        f1 = []
        f2 = []
        for individual in Vars:
            if sum(individual) == 0:
                f1.append(np.array([10000000]))
                f2.append(np.array([10000000]))
            else:
                T1 = 0
                T2 = 0
                for i in range(len(self.S_list)):
                    SX, TX = Feature_Reduce(self.S_list[i], individual), Feature_Reduce(self.T, individual)
                    T1 = T1 + mmd_rbf(SX, TX)
                    T2 = T2 + F1(SX, self.LS_list[i])
                f1.append(np.array([T1]))
                f2.append(np.array([T2]))

        f1 = np.array(f1)
        f2 = np.array(f2)
        f = np.hstack([f1, f2])
        return f

if __name__ == '__main__':
    stream = FileStream("DATA/weather.csv")
    Tx, Ty = stream.next_sample(100)
    Sx_list, Sy_list = [], []
    for i in range(3):
        Sx, Sy = stream.next_sample(100)
        Sx_list.append(Sx)
        Sy_list.append(Sy)
    n = get_dimensions(Tx)[1]
    problem = MyProblem(n=n, T=Tx, S_list=Sx_list, LS_list=Sy_list)

    # 实例化问题对象
    # 构建算法
    algorithm = ea.moea_NSGA2_templet(problem,
                                      ea.Population(Encoding='BG', NIND=50),
                                      MAXGEN=10,  # 最大进化代数
                                      logTras=0)  # 表示每隔多少代记录一次日志信息，0表示不记录。
    algorithm.mutOper.Pm = 0.2  # 修改变异算子的变异概率
    algorithm.recOper.XOVR = 0.9  # 修改交叉算子的交叉概率
    # 求解
    res = ea.optimize(algorithm, verbose=False, drawing=0, outputMsg=False, drawLog=False, saveFlag=False)['Vars']
    solution = res[0]
    for var in res:
        if sum(var) < sum(solution):
            solution = var
    print(solution)