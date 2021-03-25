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
    return result[0], result[1]


def line_fitting(x, y):
    least_squares_x, least_squares_y = least_squares(x, y)
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


def curved_line(x, y):
    popt = np.polyfit(x, y, 2)
    yn = np.polyval(popt, y)
    plt.plot(x, yn)


datafile = sys.argv[1]  # sys.argv contains the arguments passed to the program
totalX, totalY = load_points_from_file(datafile)
X = totalX[:20]
Y = totalY[:20]
resultX, resultY = line_fitting(X, Y)
pdf = calculate_pdf(Y)
print("Pdf:", pdf)
plt.plot(resultX, resultY, 'r-', lw=4)
curved_line(X, Y)
print("X: ", resultX)
print("Y: ", resultY)
Xerror = find_error(X)
Yerror = find_error(Y)
print("X error: ", Xerror)
print("Y error: ", Yerror)
# view_data_segments(resultX, resultY)
view_data_segments(X, Y)
