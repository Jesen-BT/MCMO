from sklearn.mixture import GaussianMixture
from skmultiflow.data.file_stream import FileStream
import numpy as np
import math

class DGMM(GaussianMixture):
    def __init__(
            self,
            n_components=1,
            *,
            covariance_type="full",
            tol=1e-3,
            reg_covar=1e-6,
            max_iter=100,
            n_init=1,
            init_params="kmeans",
            weights_init=None,
            means_init=None,
            precisions_init=None,
            random_state=None,
            warm_start=False,
            verbose=0,
            verbose_interval=10
    ):
        super().__init__(n_components=n_components,
            tol=tol,
            reg_covar=reg_covar,
            max_iter=max_iter,
            n_init=n_init,
            init_params=init_params,
            random_state=random_state,
            warm_start=warm_start,
            verbose=verbose,
            verbose_interval=verbose_interval,)
        self.covariance_type = covariance_type
        self.weights_init = weights_init
        self.means_init = means_init
        self.precisions_init = precisions_init

    def partial_fit(self, x, alpha):
        sp = self.predict_proba(x)

        for i in range(len(sp)):
            for j in range(len(self.weights_)):
                self.weights_[j] = (1 - alpha) * self.weights_[j] + alpha * sp[i][j]
                mean_t = self.means_[j] + (alpha * sp[i][j])/self.weights_[j] * (x[i] - self.means_[j])
                self.covariances_[j] = self.covariances_[j] - np.dot((mean_t - self.means_[j]).T, (mean_t - self.means_[j])) + (alpha * sp[i][j])/self.weights_[j] * (self.covariances_[j] - np.dot((x[i] - mean_t).T, (x[i] - mean_t)))
                self.means_[j] = mean_t
            weight_sum = sum(self.weights_)
            for j in range(len(self.weights_)):
                self.weights_[j] = self.weights_[j]/weight_sum

    def evaluation_weight(self, x):
        pro = -1./self._estimate_log_prob(x)
        weight = np.zeros(len(pro))
        for i in range(len(pro)):
            weight[i] = max(pro[i])
        return weight*1000

if __name__ == '__main__':
    stream = FileStream("DATA/weather.csv")
    data, label = stream.next_sample(500)
    stream.next_sample(5000)
    gm = DGMM(n_components=10, random_state=0).fit(data)
    ndata, y = stream.next_sample(10)
    ndata, y = stream.next_sample(5)
    print(np.std(list(gm.evaluation_weight(ndata))))
    # print(gm.partial_fit(ndata, 0.05))

