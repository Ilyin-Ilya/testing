import inline as inline
import numpy as np
import math
import pandas as pd
import scipy
from scipy import interpolate
from sklearn import datasets, linear_model
import scipy.linalg as sla
import matplotlib.pyplot as plt
from scipy import interpolate
import PySimpleGUI as sg

my_font = ("Helvetica", 13)
"  y'-2xy=0"
"  y(0) = 1"

def RepresentsInt(s:str):
    try:
        int(s)
        return True
    except ValueError:
        return False

def create_matr(N, X, step,y_init):
    matr = list(list())
    for i in range(0, N - len(y_init)):
        my_list = list()
        j=0
        while j< N:
            if j == i:
                my_list.append((1 / step)**2 + 1)

                my_list.append(-2*(1 / step)**2)

                my_list.append((1 / step)**2)
                j += 3
                continue
            my_list.append(0)
            j+=1
        matr.append(my_list)

    for i in range(0,len(y_init)):
        list2 = list()
        for j in range(0,N):
            if j==i:
                list2.append(1)
                continue
            list2.append(0)
        matr.append(list2)
    return matr


def my_grad(x_begin,x_end,y_init, order,first_field, operation, second_field, func):
    N = 500
    step = (x_end - x_begin) / N

    X = np.linspace(x_begin, x_end, N)
    y = np.zeros(N)

    matr = np.array(create_matr(N, X, step,y_init))
    right_side = np.zeros((N, 1))
    for i in range(0,N-order):
        if(func=="tgx"):
            right_side[i]=math.tan(X[i])
    for i in range(0,len(y_init)):
         right_side[N - i - 1][0] = y_init[i]
    solution = np.linalg.solve(matr, right_side)
    f=np.rot90(solution)
    f1 = scipy.interpolate.interp1d(X, f, kind='cubic')
    plt.plot(X, solution, '-')
    plt.show()

def parseEquation(equation: str):
    i=0
    order=-1
    first_field=list()
    second_field = [1,0,0]
    right_side = "0"
    while(i<len(equation)):
        while equation[i] != "+" and equation[i] != "-" and equation[i] != "=" :
            if equation[i] == "y":
                i+=1
                order+=1
                while equation[i] == "\'":
                    order += 1
                    i += 1
                first_field.append(order)
                order=0
        operation = equation[i]
        i+=1
        while equation[i]!="=":
            j=i
            f = equation[5:6]
            while RepresentsInt(equation[j:(i+1)]):
                second_field[0]=int(equation[j:(i+1)])
                i+=1
            if equation[i]=="x":
                second_field[1]=1
                i+=1
            if equation[i]=="y":
                second_field[2]=1
                i+=1
        i+=1
        if (equation[i]=="0"):
            right_side=0
        if (equation[i]=="t" and equation[i+1]=="g" and equation[i+2]=="x"):
            right_side="tgx"
        i+=1
    return first_field,operation,second_field,right_side

def getOrderOfTheEquation(equation: str):
    res = 0
    i = 0
    while i < len(equation):
        current_res = 0
        if equation[i] == "\'":
            while (equation[i] == "\'"):
                current_res += 1
                i += 1
            if (current_res > res):
                res = current_res
        i += 1
    return res


def gui(name):
    'y\'\'y-2xy=0'
    first_field, operation, second_field, right_side = parseEquation('y\'\'y-2xy=0')
    general_info = [[sg.Text('Differential equation', font=my_font), sg.InputText(key="-equation-", font=my_font)],
                    [sg.Button("Add initial conditions", key="-add_initial_conditions-", font=my_font)]]
    #     # Create the window
    window = sg.Window("DiffEquationSolver", general_info, margins=(400, 300))

    # Create an event loop
    while True:
        event, values = window.read()
        # End program if user closes window or
        # presses the OK button
        if event == sg.WIN_CLOSED:
            break
        if event == "-add_initial_conditions-":
            equation = values['-equation-']
            if (equation == ""):
                sg.popup("Please, input the equation first")
                continue
            equation_order = getOrderOfTheEquation(equation)
            window1 = sg.Window("Differential equation", add_initial_conditions(equation_order, equation),
                                margins=(400, 300))
            window.Close()
            window = window1
            continue
        if event == "-solve-":
            try:
                init_x, init_y = get_initial_conditions(values)
            except Exception:
                sg.popup("Please, fill initial conditions properly. The value can not be interpreted as a number")
                continue
            x_begin = min(init_x)
            x_end=(max(init_x)-x_begin+1)*10
            first_field, operation, second_field, right_side = parseEquation(equation)
            my_grad(x_begin,x_end,init_y,equation_order)
            continue
        if event == "-amend-":
            my_grad()
            continue

    window.close()


def add_initial_conditions(num: int, equation: str):
    general_info = [[sg.Text(equation, font=my_font)],
                    [sg.Button("Solve the equation", key="-solve-", font=my_font),
                     sg.Button("Amend equation", key="-amend-", font=my_font)]]
    order = ""
    for i in range(num):
        general_info[1].append([sg.Text("y" + order + "(", font=my_font),
                                sg.InputText(key="-coef_" + str(i) + "-", size=(5, 5), font=my_font),
                                sg.Text(") =", font=my_font),
                                sg.InputText(key="-value_" + str(i) + "-", size=(5, 5), font=my_font)])
        order += "'"
    return general_info


def get_initial_conditions(values: dict):
    coefs = []
    equals = []
    for i in range(0, int((len(values) + 1) / 2)):
        coefs.append(float(values["-coef_" + str(i) + "-"]))
        equals.append(float(values["-value_" + str(i) + "-"]))

    return coefs, equals


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
