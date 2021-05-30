import inline as inline
import numpy as np
import pandas as pd
import scipy
from sklearn import datasets, linear_model
import scipy.linalg as sla
import matplotlib.pyplot as plt
from scipy import interpolate
import PySimpleGUI as sg


def mse(preds, y):
    """
    повертає середньоквадратичну помилку між preds та y.
    """
    return ((preds - y)**2).mean()


def grad_descent(X, y, lr, num_iter=100):
    global W, b
    W = np.random.rand(X.shape[1])
    b = np.array(np.random.rand(1))

    losses = []

    N = X.shape[0]
    for iter_num in range(num_iter):
        preds = predict(X)
        losses.append(mse(preds, y))

        w_grad = np.zeros_like(W)
        b_grad = 0
        for sample, prediction, label in zip(X, preds, y):
            w_grad += 2 * (prediction - label) * sample
            b_grad += 2 * (prediction - label)

        W -= lr * w_grad
        b -= lr * b_grad
    return losses


def generate_data(range_, a, b, std, num_points=100):
    """Генерує дани, що підпорядковуються залежності y = a*x + b + е,
    де е - нормально розподілене зі стандартним відхиленням std и нульовим середнім."""
    X_train = np.random.random(num_points) * (range_[1] - range_[0]) + range_[0]
    y_train = a * X_train + b + np.random.normal(0, std, size=X_train.shape)

    return X_train, y_train

def predict(X):
    """
   Передбачує занчення y, користуючись поточні значення W та b
    """
    global W, b
    g = X @ W + b.reshape(-1, 1)
    f = np.squeeze(g)
    return np.squeeze(X @ W + b.reshape(-1, 1))

"  y'-2xy=0"
"  y(0) = 1"


def solve_weights(X, y):
    """
    Обчислює значення W,b методом найменших квадратів для X та y.
    """
    global W, b

    N = X.shape[0]
    # додаємо до ознак фіктивний розмір
    bias = np.ones((N, 1))
    X_b = np.append(bias, X, axis=1)

    W_full = np.linalg.inv(X_b.T @ X_b) @ X_b.T @ y

    W = W_full[1:]
    b = np.array([W_full[0]])

def create_matr(N, X, step):
    matr = list(list())
    for i in range(0,N-1):
        my_list=list()
        for j in range(0,N):
            if j==i:
                my_list.append(-2*X[j]-1/step)
                continue
            if j==i+1:
                my_list.append(1/step)
                continue
            my_list.append(0)
        matr.append(my_list)
    list2 = list()
    list2.append(1)
    for i in range(1,N):
        list2.append(0)
    matr.append(list2)
    return matr

def my_grad():
    x_begin = 0
    x_end = 1
    N=100
    step = (x_end-x_begin)/N

    X=np.linspace(x_begin,x_end,N)
    y=np.zeros(N)

    matr = np.array(create_matr(N,X,step))
    right_side = np.zeros((N,1))
    right_side[N-1][0]=1
    solution = np.linalg.solve(matr, right_side)

    plt.plot(X, solution,'o')
    plt.show()

    losses = []

    N = X.shape[0]
    for iter_num in range(15000):
        preds = predict(X)
        losses.append(mse(preds, y))

        w_grad = np.zeros_like(W)
        b_grad = 0
        for sample, prediction, label in zip(X, preds, y):
            w_grad += 2 * (prediction - label) * sample
            b_grad += 2 * (prediction - label)

        W -= 0.05 * w_grad
        b -= 0.05 * b_grad


def gui(name):
    general_info = [[sg.Text('Differential equation'),sg.InputText(key="-equation-")],
                    [sg.Button("Solve the equation", key="-solve-")]]
    # Create the window
    window = sg.Window("DiffEquationSolver", general_info, margins=(400, 300))

    # Create an event loop
    while True:
        event, values = window.read()
        # End program if user closes window or
        # presses the OK button
        if event == sg.WIN_CLOSED:
            break
        if event == "-solve-":
            number_of_params = values['-equation-']
            my_grad()
    window.close()
# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    gui("")
    """
    real_a = 0.34
    real_b = 13.7
    real_std = 7

    # Згенеруємо дані на проміжку від 0 до 150
    X_train, y_train = generate_data([0, 150], real_a, real_b, real_std)

    pd.DataFrame({'X': X_train, 'Y': y_train}).head()

    plt.scatter(X_train, y_train, c='black')
    plt.plot(X_train, 0.34*X_train+13.7)
    plt.show()

    losses = grad_descent(X_train.reshape(-1, 1), y_train, 1e-9, 15000)
    plt.plot(losses), losses[-1]
    plt.show()

    plt.scatter(X_train, y_train, c='r')
    plt.plot(X_train, real_a * X_train + real_b)
    plt.plot(X_train, np.squeeze(X_train.reshape(-1, 1) @ W + b.reshape(-1, 1)))
    plt.show()
    """
# See PyCharm help at https://www.jetbrains.com/help/pycharm/
