import array
from math import sqrt, exp

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def cross_entropy(x: np.array, y: np.array) -> float:
    return -np.sum(y * np.log(x))


def der_cross_entropy(x: np.array, y: np.array) -> np.array:
    eps = 0.001
    return - x / (y + eps)


def softmax(x: np.array) -> np.array:
    xx = x - np.max(x)
    return np.exp(xx) / np.sum(np.exp(xx))


def der_softmax(x: np.array) -> np.array:
    size = x.size
    f = softmax(x)
    jacobian = np.zeros((size, size))
    for i in range(size):
        for j in range(size):
            jacobian[i][j] = f[i]*(1 - f[j]) if i == j else -f[i]*f[j]
    return jacobian


def tanh(x: np.array) -> np.array:
    return 2 * sigmoid(2*x) - 1


def der_tanh(x: np.array) -> np.array:
    4 * der_sigmoid(2*x)


def relu(x: np.array) -> np.array:
    shape = x.shape
    xf = x.flatten()
    return np.array([c if c > 0 else exp(c) - 1 for c in xf]).reshape(shape)


def der_relu(x: np.array) -> np.array:
    shape = x.shape
    xf = x.flatten()
    return np.array([1 if c > 0 else exp(c) for c in xf]).reshape(shape)


def sigmoid(x: np.array) -> np.array:
    return 1 / (1 + np.exp(-x))


def der_sigmoid(x: np.array) -> np.array:
    return np.array([sigmoid(c) * (1 - sigmoid(c)) for c in x])


def mse(x: np.array, y: np.array) -> np.array:
    return np.power(x - y, 2)


def der_mse(x: np.array, y: np.array) -> np.array:
    return 2 * (x - y)


def derivative(f: callable) -> callable:
    if f == tanh:
        return der_tanh
    if f == softmax:
        return der_softmax
    if f == relu:
        return der_relu
    if f == sigmoid:
        return der_sigmoid
    if f == mse:
        return der_mse
    if f == cross_entropy:
        return der_cross_entropy


class Layer:
    def forward_prop(self, x: np.array) -> np.array:
        pass

    def back_prop(self, grad: np.array) -> np.array:
        pass


class Dense(Layer):
    def __init__(self, shape: tuple[int, int], activation: callable):
        self.activation = activation
        self.W = np.random.rand(shape[0], shape[1]) * 2 - 1
        self.b = np.random.rand(shape[0]) * 2 - 1
        self.a = None
        self.input = None
        self.eta = 0.2

    def forward_prop(self, x: np.array) -> np.array:
        self.input = x
        print(x)
        self.a = np.dot(self.W, x) + self.b
        return self.activation(self.a)

    def back_prop(self, grad: np.array) -> np.array:
        local_grad = np.dot(derivative(self.activation)(self.a), grad)
        self.b -= self.eta * local_grad
        gr_W = np.outer(local_grad, np.transpose(self.input))
        local_grad = np.dot(np.transpose(self.W), local_grad)
        self.W -= self.eta * gr_W
        return local_grad


def convolution(map: np.array, kernel: np.array) -> np.array:
    in_a, in_b = map.shape
    kernel_size = kernel.shape[0]
    out_a = in_a - kernel_size + 1
    out_b = in_b - kernel_size + 1
    out = np.zeros((out_a, out_b))
    for i in range(out_a):
        for j in range(out_b):
            out[i, j] = np.sum(map[i:i + kernel_size, j:j + kernel_size] * kernel)
    return out


def fulling(x: np.array, new_size: int) -> np.array:
    vert = (new_size - x.shape[0]) // 2
    z = np.zeros((vert, x.shape[1]))
    res = np.vstack((z, x, z))
    hor = (new_size - x.shape[1]) // 2
    z = np.zeros((new_size, hor))
    res = np.hstack((z, res, z))
    return res


class Conv(Layer):
    def __init__(self, size: int, activation: callable):
        self.size = size
        self.eta = 0.1
        self.W = np.ones((size, size))
        self.activation = activation
        self.input = None
        self.h = None

    def forward_prop(self, x: np.array) -> np.array:
        self.input = x
        out = convolution(x, self.W)
        self.h = out
        return self.activation(out)

    def back_prop(self, grad: np.array) -> np.array:
        local_grad = grad * derivative(self.activation)(self.h)
        grad_W = convolution(self.input, local_grad)
        rotate_W = np.rot90(self.W, 2)
        grad_size = self.input.shape[0] + grad.shape[0] - 1
        local_grad = convolution(fulling(rotate_W, grad_size), local_grad)
        #self.W -= self.eta * grad_W
        #print(self.W)
        return local_grad


class AvgPooling(Layer):
    def __init__(self, size: int):
        self.size = size

    def forward_prop(self, x: np.array) -> np.array:
        in_a, in_b = x.shape
        out_a = in_a // self.size
        out_b = in_b // self.size
        out = np.zeros((out_a, out_b))
        for i in range(0, out_a, self.size):
            for j in range(0, out_b, self.size):
                out[i, j] = np.mean(x[i:i + self.size, j:j + self.size])
        return out

    def back_prop(self, grad: np.array) -> np.array:
        k_grad = (1 / (self.size * self.size)) * grad
        local_grad = k_grad.copy()
        for i in range(self.size - 1):
            local_grad = np.vstack((local_grad, k_grad))
        col_grad = local_grad.copy()
        for i in range(self.size - 1):
            local_grad = np.hstack((local_grad, col_grad))
        return local_grad


class Flatten(Layer):
    def forward_prop(self, x: np.array) -> np.array:
        return x.flatten()

    def back_prop(self, grad: np.array) -> np.array:
        size = int(sqrt(grad.size))
        return grad.reshape((size, size))


class Model:
    def __init__(self, loss_func: callable):
        self.layers = []
        self.loss_func = loss_func

    def add(self, layer: Layer) -> None:
        self.layers.append(layer)

    def fit(self, x: np.array, y: np.array, epochs: int) -> None:
        for epoch in range(epochs):
            errors = []
            for i, row in enumerate(x):
                res = row
                for layer in self.layers:
                    res = layer.forward_prop(res)
                errors.append(res)
                grad = derivative(self.loss_func)(res, y[i])
                norm = np.linalg.norm(grad, ord=1)
                #grad = grad if norm < 1 else (1 / norm) * grad
                for k in range(len(self.layers) - 1, -1, -1):
                    grad = self.layers[k].back_prop(grad)
                    norm = np.linalg.norm(grad, ord=1)
                    #grad = grad if norm < 1 else (1 / norm) * grad
                print(f"res: {res}\nlabel: {y[i]}\nerror: {self.loss_func(res, y[i])}\nnorm_grad: {np.linalg.norm(grad, ord=1)}")
            print(f"loss after {epoch}: {self.loss_func(np.array(errors), y) / len(y)}")

    def predict(self, samples: pd.DataFrame) -> np.array:
        res = []
        s = samples.to_numpy()
        for sample in s:
            r = sample
            for layer in self.layers:
                r = layer.forward_prop(r)
            res.append(r)
        return res


def precision_recall(predicts: array, labels: pd.DataFrame, threshold: float) -> (float, float, float, float):
    preds = [1 if c[0] > threshold else 0 for c in predicts]
    tp = sum([l == 1 and preds[i] == 1 for i, l in enumerate(labels)])
    fp = sum([l == 0 and preds[i] == 1 for i, l in enumerate(labels)])
    tn = sum([l == 0 and preds[i] == 0 for i, l in enumerate(labels)])
    fn = sum([l == 1 and preds[i] == 0 for i, l in enumerate(labels)])
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    fpr = fp / (fp + tn)
    acc = (tp + tn) / (tp + tn + fp + fn)
    return precision, recall, fpr, acc


def paint(x: array, y: array) -> None:
    plt.plot(x, y)
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.show()


def rewrite_samples(samples: pd.DataFrame, size: int) -> np.array:
    arr = []
    s = samples.to_numpy()
    for x in s:
        arr.append(x.reshape(size, size))
    return arr


def rewrite_labels(y: pd.DataFrame) -> np.array:
    labels = y.to_numpy()
    res = []
    for label in labels:
        n = np.zeros(10)
        n[label[0]] = 1
        res.append(n)
    return res


def classification():
    targets = ['label']
    data = pd.read_csv("C:\\Users\\User\\PycharmProjects\\lab3\\mnist_train.csv")
    x_train = data.iloc[:, ~data.columns.isin(targets)]
    y_train = data.iloc[:, data.columns.isin(targets)]
    x_train = rewrite_samples(x_train / 256, 28)
    y_train = rewrite_labels(y_train)
    data = pd.read_csv("C:\\Users\\User\\PycharmProjects\\lab3\\mnist_test.csv")
    x_test = data.iloc[:, ~data.columns.isin(targets)]
    x_test = rewrite_samples(x_test / 256, 28)
    y_test = data.iloc[:, data.columns.isin(targets)]
    y_test = rewrite_labels(y_test)

    model = Model(cross_entropy)
    model.add(Conv(5, sigmoid))
    model.add(AvgPooling(2))
    model.add(Conv(5, sigmoid))
    model.add(AvgPooling(2))
    model.add(Flatten())
    #model.add(Dense((12, 16), relu))
    model.add(Dense((10, 16), softmax))
    model.fit(x_train, y_train, 1)

    '''thresholds = [i / 20 for i in range(20)]
    predicts = model.predict(x_test)
    precisions = []
    recalls = []
    fprs = []
    for t in thresholds:
        pr = precision_recall(predicts, y_test[0].to_numpy(), t)
        precisions.append(pr[0])
        recalls.append(pr[1])
        fprs.append(pr[2])
        print(f'threshold {t}: acc = {pr[3]}, precision = {pr[0]}, recall = {pr[1]}, fpr = {pr[2]}')
    paint(recalls, precisions)
    paint(fprs, recalls)'''


classification()
