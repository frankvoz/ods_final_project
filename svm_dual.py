import time
import math
import numpy as np

from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from datasets import cmap
from datasets import ALPHA_CH


# constants
MAX_ITERS = 500
EPS = 1e-6


def get_A(X, y, c=0.1):
    n = X.shape[0]

    # add bias
    Xc = np.copy(X)
    Xc = np.column_stack((Xc, np.ones(n)))
    
    Z = np.multiply(y.reshape(n, 1), Xc)
    I = 1 / math.sqrt(c) * np.eye(n)

    A = np.concatenate((Z.T, I), axis=0)
    return A


def _init_x(n):
    x = np.ones(n) / n
    return x


def _objective_function(A, x):
    loss = np.linalg.norm(np.dot(A, x)) ** 2
    return loss


def _calculate_gradient(A, x):
    prod = np.dot(A, x)
    grad = 2 * np.dot(A.T, prod)
    return grad


def _get_fw_solution(grad, size):
    idx = np.argmin(grad)
    s = np.zeros(size)
    s[idx] = 1

    return s


def _exact_line_search(A, alpha_guess, x_k, d_k):
        # alpha_guess: initial guess for alpha
        # x_k: current value
        # d_k: search direction

        epsilon = 1e-6

        def phi(alpha):
            x_next = x_k + alpha * d_k
            return _objective_function(A, x_next)
        
        alpha_low = 0
        alpha_high = alpha_guess

        # golden ratio search
        # TODO: study
        golden_ratio = (math.sqrt(5) - 1) / 2

        while abs(alpha_high - alpha_low) > epsilon:
            alpha1 = alpha_low + (1 - golden_ratio) * (alpha_high - alpha_low)
            alpha2 = alpha_low + golden_ratio * (alpha_high - alpha_low)

            if phi(alpha1) < phi(alpha2):
                alpha_high = alpha2
            else:
                alpha_low = alpha1

        optimal_alpha = (alpha_low + alpha_high) / 2

        return optimal_alpha


def fit_frank_wolfe(X, y, c, epochs=MAX_ITERS, tol=EPS, search_direction='fixed', print_loss=True):
    n = X.shape[0]
    loss = []
    
    A = get_A(X, y, c)
    x = _init_x(n)

    l = _objective_function(A, x)
    prev_l = 0

    start_time = time.time()

    for iter in range(epochs):
        grad = _calculate_gradient(A, x)

        s = _get_fw_solution(grad, n)

        if search_direction == 'exact':
            step_size = _exact_line_search(A, 1.0, x, (s - x))
        else:
            step_size = 2 / ((iter + 1) + 2)

        x = (1 - step_size) * x + step_size * s

        prev_l = l
        l = _objective_function(A, x)
        loss.append(l)
        if print_loss:
            print(f"step\t{iter+1}:\t{l}")

        # TODO: find a better condition
        rel_loss = np.abs(l - prev_l)/prev_l

        '''
        if rel_loss < tol:
            break
        '''
        
    end_time = time.time()
    ttime = end_time - start_time # total time for computation

    return A, x, loss, ttime


def fit_pairwise_fw(X, y, c, epochs=MAX_ITERS, tol=EPS, print_loss=True):
    n = X.shape[0]
    loss = []

    A = get_A(X, y, c)
    x = _init_x(n)

    l = _objective_function(A, x)
    prev_l = 0

    alpha = 1

    start_time = time.time()

    for iter in range(epochs):
        grad = _calculate_gradient(A, x)
        s_fw = _get_fw_solution(grad, n)

        # calculate away-step
        idx = np.argmax(grad)
        s_as = np.zeros(n)
        s_as[idx] = 1

        # calculate descent direction
        d = s_fw - s_as

        # calculate step-size
        gamma = _exact_line_search(A, alpha, x, d)
        alpha = alpha - gamma
        x += gamma * d

        prev_l = l
        l = _objective_function(A, x)
        loss.append(l)
        if print_loss:
            print(f"step\t{iter+1}:\t{l}")

        rel_loss = np.abs(l - prev_l)/prev_l

        '''
        if rel_loss < tol:
            break
        '''
        
    end_time = time.time()
    ttime = end_time - start_time

    return A, x, loss, ttime


def predict(w, b, X):
    y_pred = np.sign(np.dot(X, w) + b)
    return y_pred


def get_margin(A, x_hat):
    d = A.shape[0] - A.shape[1] - 1

    res = np.dot(A, x_hat)

    w = res[:d]
    b = res[d]
    xi = res[d+1:] # slack variables
    return w, b, xi


def plot_loss(loss, num_iter=None):
    if num_iter is None:
        num_iter = len(loss)
    xp = np.arange(1, num_iter+1)
    plt.grid()
    plt.plot(xp, loss[:num_iter])
    plt.show()


def plot_cmat(y, y_pred):
    labels = np.array([-1,  1])
    cm = confusion_matrix(y, y_pred, labels=labels)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(cmap=plt.cm.Blues)


def grid_search(fun, X_train, y_train, X_val, y_val):
    C = [1e-3, 1e-2, 0.1, 1, 10]
    acc_l = []

    for c in C:
        A, x_hat, _, _ = fun(X_train, y_train, c, print_loss=False)

        w, b, _ = get_margin(A, x_hat)
        y_pred = predict(w, b, X_val)
        acc = accuracy_score(y_pred, y_val)
        acc_l.append(acc)

    idx = np.argmax(acc_l)
    return C[idx]


def plot_decision_boundary(w, b, X_train, y_train, X_test, y_test, plot_train=False, plot_test=True):
    def get_hyperplane(x, w, b, offset):
        return (-w[0] * x - b + offset) / w[1]
    
    _, ax = plt.subplots(1, 1, figsize=(10,6))

    if plot_train:
        plt.scatter(X_train[:, 0], X_train[:, 1], marker='o', c=y_train, cmap=cmap, s=100, alpha=ALPHA_CH)
    if plot_test:
        marker = 'o'
        if plot_train:
            marker = 'x'
        plt.scatter(X_test[:, 0], X_test[:, 1], marker=marker, c=y_test, cmap=cmap, s=100, alpha=ALPHA_CH)

    x0_1 = np.amin(X_train[:, 0])
    x0_2 = np.amax(X_train[:, 0])

    x1_1 = get_hyperplane(x0_1, w, b, 0)
    x1_2 = get_hyperplane(x0_2, w, b, 0)

    x1_1_m = get_hyperplane(x0_1, w, b, -1)
    x1_2_m = get_hyperplane(x0_2, w, b, -1)

    x1_1_p = get_hyperplane(x0_1, w, b, 1)
    x1_2_p = get_hyperplane(x0_2, w, b, 1)

    ax.plot([x0_1, x0_2], [x1_1, x1_2], "-", c='k', lw=1, alpha=0.9)
    ax.plot([x0_1, x0_2], [x1_1_m, x1_2_m], "--", c='grey', lw=1, alpha=0.8)
    ax.plot([x0_1, x0_2], [x1_1_p, x1_2_p], "--", c='grey', lw=1, alpha=0.8)

    x1_min = np.amin(X_train[:, 1])
    x1_max = np.amax(X_train[:, 1])
    ax.set_ylim([x1_min - 3, x1_max + 3])

    for spine in ['top','right']:
        ax.spines[spine].set_visible(False)

    plt.show()