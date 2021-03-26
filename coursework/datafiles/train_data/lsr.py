import sys

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


# no longer using
def line_fitting(x, y):
    least_square = least_squares(x, y)
    least_squares_x, least_squares_y = least_square[0], least_square[1]
    print("Least Squares X:", least_squares_x)
    print("Least Squares Y:", least_squares_y)
    x_endpoint_min = x.min()
    x_endpoint_max = x.max()
    y_endpoint_min = least_squares_x + least_squares_y * x_endpoint_min
    y_endpoint_max = least_squares_x + least_squares_y * x_endpoint_max
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


def curved_line():
    resized_x = quadratic_resizer(X)
    matrix = least_squares_formula(resized_x, Y)
    # print("matrix: ", matrix)
    xs = np.linspace(X[0], X[19], 20)
    ys = quadratic_resizer(xs) @ matrix  # @ is matrix multiplication
    return xs, ys


def calculate_pdf(data):
    fraction = 1 / (np.sqrt(np.var(data)) * np.sqrt(2 * np.pi))
    exponent = - ((np.power((data - np.mean(data)), 2)) / (2 * np.var(data)))
    result = fraction * np.power(np.e, exponent)
    return result


# the next step is probably this with cross validation
def find_error(y_estimates):
    error = np.sum((y_estimates - Y)**2)
    return error


def run_calculations():
    # resultX, resultY = line_fitting(X, Y)
    xs, ys = curved_line()
    # plt.plot(resultX, resultY, 'y-', lw=4)
    plt.plot(xs, ys, 'r-', lw=4)  # @ is matrix multiplication
    # pdf = calculate_pdf(Y)
    # print("Pdf:", pdf)
    # print("X: ", resultX)
    # print("Y: ", resultY)
    error = find_error(ys)
    print("Error: ", error)
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
