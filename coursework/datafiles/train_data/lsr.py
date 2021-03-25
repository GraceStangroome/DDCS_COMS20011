import sys

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


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


def calculate_pdf(data):
    fraction = 1 / (np.sqrt(np.var(data)) * np.sqrt(2 * np.pi))
    exponent = - ((np.power((data - np.mean(data)), 2)) / (2 * np.var(data)))
    result = fraction * np.power(np.e, exponent)
    return result


def least_squares(x, y):
    ones = np.ones(x.shape)
    x_with_ones = np.column_stack((ones, x))
    result = np.linalg.inv(x_with_ones.T.dot(x_with_ones)).dot(x_with_ones.T).dot(y)
    return result


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


def find_error(data):
    error = np.var(data) * (len(data) - 1)
    return error


def quadratic_resizer(x):
    return np.column_stack((np.ones(x.shape), x, x**2))


def curved_line(x, y):
    matrix = least_squares(x, y)
    print(matrix)
    sigma = 0  # output noise
    quad_term = quadratic_resizer(x)
    linear_term = -1
    bias = 0
    print(np.array(matrix))
    part1 = np.array(quad_term) * np.array(matrix)
    part2 = part1 ** 2
    part3 = part2 + linear_term * matrix[0] + bias + sigma * y
    return part3


datafile = sys.argv[1]  # sys.argv contains the arguments passed to the program
totalX, totalY = load_points_from_file(datafile)
start = 0
end = 20
while end < len(totalX):  # data is in chunks of 20
    X = totalX[start:end]
    Y = totalY[start:end]
    print(X)
    resultX, resultY = line_fitting(X, Y)
    line = curved_line(X, Y)
    pdf = calculate_pdf(Y)
    print("Pdf:", pdf)
    plt.plot(resultX, resultY, 'y-', lw=4)
    plt.plot(line, 'm-', lw=4)
    print("X: ", resultX)
    print("Y: ", resultY)
    Xerror = find_error(X)
    Yerror = find_error(Y)
    print("X error: ", Xerror)
    print("Y error: ", Yerror)
    # view_data_segments(resultX, resultY)
    view_data_segments(X, Y)
    start += 20
    end += 20
