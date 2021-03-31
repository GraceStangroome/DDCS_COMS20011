import random
import sys

import numpy as np
import pandas as pd
from numpy import polynomial
from matplotlib import pyplot as plt


# no longer using
def line_fitting(x, y):
    least_square = least_squares(x, y)
    least_squares_x, least_squares_y = least_square[0], least_square[1]
    print("Least Squares X:", least_squares_x)
    print("Least Squares Y:", least_squares_y)
    x_endpoint_min = x.min()
    x_endpoint_max = x.max()
    y_endpoint_min = np.sin(least_squares_x + least_squares_y * x_endpoint_min)
    y_endpoint_max = np.sin(least_squares_x + least_squares_y * x_endpoint_max)
    return [x_endpoint_min, x_endpoint_max], [y_endpoint_min, y_endpoint_max]


def load_points_from_file(filename):
    """Loads 2d points from a csv called filename
    Args:
        filename : Path to .csv file
    Returns:
        (xs, ys) where xs and ys are a numpy array of the co-ordinates.
    """
    points = pd.read_csv(filename, header=None)
    return points[0].values, points[1].values


def view_data_segments(xs, ys):
    """Visualises the input file with each segment plotted in a different colour.
    Args:
        xs : List/array-like of x co-ordinates.
        ys : List/array-like of y co-ordinates.
    Returns:
        None
    """
    assert len(xs) == len(ys)
    assert len(xs) % 20 == 0
    len_data = len(xs)
    num_segments = len_data // 20
    colour = np.concatenate([[i] * 20 for i in range(num_segments)])
    plt.set_cmap('Dark2')
    plt.scatter(xs, ys, c=colour)
    plt.show()


def least_squares_formula(x, y):
    return np.linalg.inv(x.T.dot(x)).dot(x.T).dot(y)


def least_squares(x, y):
    ones = np.ones(x.shape)
    x_with_ones = np.column_stack((ones, x))
    return least_squares_formula(x_with_ones, y)


def quadratic_resizer(x):
    # for q in range(a):
    # np.stack()
    # 2d arrray with 1 array of 1s
    # then append the set times the postition in the loop
    return np.column_stack((np.ones(x.shape), x, x ** 2, x ** 3))


def curved_line(xs, ys):
    resized_x = quadratic_resizer(xs)
    matrix = least_squares_formula(resized_x, ys)
    # print("matrix: ", matrix)
    resulting_x = np.linspace(X[0], X[19], 20)  # uses X over 20 so that it covers all the data
    resulting_y = quadratic_resizer(resulting_x) @ matrix  # @ is matrix multiresized_xplication
    unknown_function = line_fitting(xs, ys)
    return resulting_x, resulting_y, unknown_function


def calculate_pdf(data):
    fraction = 1 / (np.sqrt(np.var(data)) * np.sqrt(2 * np.pi))
    exponent = - ((np.power((data - np.mean(data)), 2)) / (2 * np.var(data)))
    result = fraction * np.power(np.e, exponent)
    return result


# the next step is probably this with cross validation
def find_error(y_estimates, y_test):
    error = 0
    for i in range(len(y_test)):
        for j in range(len(y_estimates)):
            error += ((y_estimates[j] - y_test[i]) ** 2)
    error = error / 4
    return error


def cheb_formula(x, c):
    # c is int
    coefs = c * [0] + [1]   # what is this notation
    return np.polynomial.chebyshev.chebval(x, coefs)


def chebyshev(data_x, order):
    # assert (-1 <= data).all() and (data <= 1).all()
    xs = []
    for c in range(order):
        xs.append(cheb_formula(data_x, c))
        print(xs)
    return np.concatenate(xs, np.ones(len(xs)))


def cross_validation(ys, y_test):
    error = find_error(ys, y_test)
    return error


def run_calculations():
    # resultX, resultY = line_fitting(X, Y)
    # Using 80% of the data for training
    x_train, y_train = np.empty(16), np.empty(16)
    x_test, y_test = np.empty(4), np.empty(4)
    options = [i for i in range(0, 20)]  # using X for no reason, could be Y, but X + Y should be the same length
    # randomly choosing 16 elements of X and Y for training
    for i in range(16):
        for_training = random.choice(options)
        x_train[i] = X[for_training]
        y_train[i] = Y[for_training]
        options.remove(for_training)
    # the remaining options are for testing
    for i in range(4):
        x_test[i] = X[options[i]]
        y_test[i] = Y[options[i]]

    xs, ys, unknown_function = curved_line(x_train, y_train)
    # plt.plot(resultX, resultY, 'y-', lw=4)
    plt.plot(xs, ys, 'r-', lw=4)
    plt.plot(unknown_function, 'y-', lw=4)
    # pdf = calculate_pdf(Y)
    # print("Pdf:", pdf)
    # print("X: ", resultX)
    # print("Y: ", resultY)
    error = find_error(ys, y_test)
    # cross_val = cross_validation(ys, y_test)
    print("Error: ", error)
    # print("Cross validation: ", cross_val)
    # view_data_segments(resultX, resultY)


datafile = sys.argv[1]  # sys.argv contains the arguments passed to the program
totalX, totalY = load_points_from_file(datafile)
start = 0
end = 20
while end <= len(totalX):  # data is in chunks of 20
    print("Start: ", start, " End: ", end)
    X = totalX[start:end]
    Y = totalY[start:end]
    run_calculations()
    view_data_segments(X, Y)
    start += 20
    end += 20
print("Finished")
