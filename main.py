import os
import shutil

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

OUT_FOLDER = None
RESULT_FILE = "results.txt"


def set_output(dir_name="out"):
    global OUT_FOLDER
    if os.path.exists(dir_name):
        shutil.rmtree(dir_name)
    os.makedirs(dir_name)
    OUT_FOLDER = dir_name


def write_out(file_name, data, mode="a"):
    if OUT_FOLDER is None:
        return
    dest = os.path.join(OUT_FOLDER, file_name)
    with open(dest, mode) as out:
        out.write(data + '\n')


def write_out_fig(name):
    if OUT_FOLDER is None:
        return
    dest = os.path.join(OUT_FOLDER, name)
    plt.savefig(dest)


class MyLinearRegression(object):
    def __init__(self, **kwargs):
        self.sample_weight = None

        # method handling
        method = kwargs.pop("method", "alg")
        allowed_methods = ("alg", "grad")
        if method not in allowed_methods:
            raise ValueError(
                "Method should be among {}, while {} is provided".format(
                    allowed_methods, method
                ))
        self.method = method
        self.params = kwargs

    def _fit_alg(self, X, y):
        X_t = np.transpose(X)
        self.sample_weight = \
            np.dot(np.dot(np.linalg.inv(np.dot(X_t, X)), X_t), y)

    def _fit_grad(self, X, y):
        weight = np.zeros(X.shape[1])
        params_alpha = self.params.get('alpha', None)
        params_stop = self.params.get('stop', 0.0000001)
        params_iters = self.params.get('iters', 1000)
        for it in range(params_iters):
            new_weight = weight.copy()
            alpha = 2.0 / (it + 1) if params_alpha is None else params_alpha
            for j in range(len(new_weight)):
                grad = 0
                for i in range(X.shape[0]):
                    grad += (np.dot(weight, X[i]) - y[i]) * X[i][j]
                new_weight[j] -= alpha * 2 / X.shape[0] * grad
            if sum(x**2 for x in new_weight-weight) < params_stop:
                break
            weight = new_weight.copy()
        self.sample_weight = weight

    def fit(self, X, y):
        if X.shape[0] != y.shape[0]:
            raise ValueError("Wrong shapes of X: {} and y: {}".format(
                X.shape, y.shape))
        if self.method == "alg":
            self._fit_alg(X, y)
        elif self.method == "grad":
            self._fit_grad(X, y)

        return self.sample_weight

    def predict(self, X, sample_weight=None):
        if self.sample_weight is None and sample_weight is None:
            raise ValueError("Sample weights are not fitted")
        sample_weight = sample_weight or self.sample_weight
        if X.shape[1] != sample_weight.shape[0]:
            raise ValueError(
                "Wrong shapes of X: {} and sample weights: {}".format(
                    X.shape, sample_weight))
        return np.dot(X, sample_weight)

    def score_r2(self, X, y, sample_weight=None):
        """Return R**2 score"""
        f = self.predict(X, sample_weight)
        y_mean = np.mean(y)
        ss_tot = sum((y_value - y_mean)**2 for y_value in y)
        ss_res = sum((y[i] - f[i])**2 for i in range(len(y)))

        return 1 - ss_res / ss_tot

    def score_rss(self, X, y, sample_weight=None):
        """Return RSS score"""
        f = self.predict(X, sample_weight)
        return sum((y[i] - f[i])**2 for i in range(len(f))) / len(f)

    def score(self, X, y, sample_weight=None, method="r2"):
        if method == "r2":
            return self.score_r2(X, y, sample_weight)
        elif method == "rss":
            return self.score_rss(X, y, sample_weight)
        else:
            raise ValueError("Wrong score method {}".format(method))


def get_xy(data, y_name=None):
    if y_name is None:
        y = data.iloc[:, -1]
        X = data.iloc[:, :-1]
    else:
        y = data[y_name]
        X = data.drop(y_name, axis=1)
    return X.as_matrix(), y.as_matrix()


def my_train_test_split(*array, **options):
    random_state = options.get('random_state', None)
    test_size = options.get('test_size', 0.25)
    train_size = options.get('train_size', None)

    # check train and test sizes
    if train_size is None:
        train_size = 1.0 - test_size
    elif train_size + test_size > 1.0:
        raise ValueError("train_size + test_size should not be "
                         "greater than 1")
    if not (0.0 <= test_size <= 1.0 and 0.0 <= train_size <= 1.0):
        raise ValueError("train_size and test_size should be less than 0"
                         " or greater than 1")

    # check arrays sizes
    if len(set(arr.shape[0] for arr in array)) != 1:
        raise ValueError("arrays length should be equal")

    # set seed if provided to return the same shuffles each time
    if random_state is not None:
        np.random.seed(random_state)
    p = np.random.permutation(array[0].shape[0])

    res = []
    for arr in array:
        true_train_size = int(arr.shape[0] * train_size)
        shuffled_train = arr[p][:true_train_size]
        shuffled_test = arr[p][true_train_size:]
        res.extend((shuffled_train, shuffled_test))
    return res


def correlation_report(data, target_name):
    corr = data.corr()

    # Plot correlations between target and features
    target_corr = corr[target_name].drop(target_name)
    names = target_corr.index
    y_values = target_corr.values
    x_values = np.arange(len(names))
    plt.bar(x_values, y_values)
    plt.xticks(x_values + 0.5, names)
    plt.ylabel('Correlation coefficient')
    plt.title('Correlation coefficient between target and features')
    write_out_fig("correlation_target_others.png")

    # Build target dependency graphics
    nrows, ncols = 4, 4
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols)
    y_values = data[target_name]
    for ind, column_name in enumerate(data.columns.values):
        row, col = ind // nrows, ind % ncols
        axes[row, col].plot(data[column_name], y_values, '.')
        axes[row, col].set_xlabel(column_name)
        axes[row, col].set_ylabel(target_name)
    fig.tight_layout(h_pad=0.3)
    write_out_fig("target_others.png")

    # Print correlation matrix
    write_out(RESULT_FILE, "Correlation matrix\n{}".format(corr))


def analytic_report(X_train, X_test, y_train, y_test):
    l1 = LinearRegression()
    l2 = MyLinearRegression()
    l1.fit(X_train, y_train)
    l2.fit(X_train, y_train)
    write_out(RESULT_FILE,
              "Sklearn LinRegr R-squared score: "
              "{}".format(l1.score(X_test, y_test)))
    write_out(RESULT_FILE,
              "Custom analytic LinRegr R-squared score: "
              "{}".format(l2.score(X_test, y_test)))
    write_out(RESULT_FILE,
              "Custom analytic LinRegr RSS score: "
              "{}".format(l2.score(X_test, y_test, method="rss")))
    write_out(RESULT_FILE,
              "Custom analytic LinRegr RSS score for training data: "
              "{}".format(l2.score(X_train, y_train, method="rss")))
    write_out(RESULT_FILE,
              "Custom analytic Weights vector: "
              "{}".format(l2.sample_weight))


def gradient_report(X_train, X_test, y_train, y_test):
    lr = MyLinearRegression(method="grad")
    lr.fit(X_train, y_train)
    write_out(RESULT_FILE,
              "Custom gradient LinRegr R-squared score: "
              "{}".format(lr.score(X_test, y_test)))
    write_out(RESULT_FILE,
              "Custom gradient LinRegr RSS score: "
              "{}".format(lr.score(X_test, y_test, method="rss")))
    write_out(RESULT_FILE,
              "Custom gradient Weights vector: "
              "{}".format(lr.sample_weight))


def main():
    set_output()
    path, target_name = "housing.data", "MEDV"
    data = pd.read_csv(path, sep='\s+')
    # data normalization
    data = (data - data.mean()) / data.std()
    correlation_report(data, target_name)
    X, y = get_xy(data, y_name="MEDV")
    X_train, X_test, y_train, y_test = \
        my_train_test_split(X, y, random_state=0)
    analytic_report(X_train, X_test, y_train, y_test)
    gradient_report(X_train, X_test, y_train, y_test)

if __name__ == '__main__':
    main()
