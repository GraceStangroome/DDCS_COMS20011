import random
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


def least_squares_formula(x, y):
    return np.linalg.inv(x.T.dot(x)).dot(x.T).dot(y)


def unknown_resizer(x):
    return np.column_stack((x, np.ones(x.shape)))


def exp_line(xs, ys):
    calculated_xs = np.exp(xs)
    bias = unknown_resizer(calculated_xs)
    coefficients = least_squares_formula(bias, ys)
    calculated_ys = bias @ coefficients
    plt.plot(xs, calculated_ys, c=tableau20[0], lw=1, label='exponential')
    plt.legend()
    return xs, calculated_ys, coefficients


def tan_line(xs, ys):
    calculated_xs = np.tan(xs)
    bias = unknown_resizer(calculated_xs)
    coefficients = least_squares_formula(bias, ys)
    calculated_ys = bias @ coefficients
    plt.plot(xs, calculated_ys, c=tableau20[6], lw=1, label='tan')
    plt.legend()
    return xs, calculated_ys, coefficients


def cos_line(xs, ys):
    calculated_xs = np.cos(xs)
    bias = unknown_resizer(calculated_xs)
    coefficients = least_squares_formula(bias, ys)
    calculated_ys = bias @ coefficients
    plt.plot(xs, calculated_ys, c=tableau20[12], lw=1, label='cos')
    plt.legend()
    return xs, calculated_ys, coefficients


def sin_line(xs, ys):
    calculated_xs = np.sin(xs)
    bias = unknown_resizer(calculated_xs)
    coefficients = least_squares_formula(bias, ys)
    calculated_ys = bias @ coefficients
    plt.plot(xs, calculated_ys, c=tableau20[19], lw=1, label='sin')
    plt.legend()
    return xs, calculated_ys, coefficients


def squared_error(estimated_y, test_y):
    this_error = 0
    for i in range(len(estimated_y)):
        squared_difference = (estimated_y[i] - test_y[i]) ** 2
        this_error += squared_difference
    return this_error


def find_exp_error(coeffs, x_test, y_test):
    y_estimates = []
    for x in x_test:
        y_estimates.append(coeffs[0] * np.exp(x) + coeffs[1])
    return squared_error(y_estimates, y_test)


def find_tan_error(coeffs, x_test, y_test):
    y_estimates = []
    for x in x_test:
        y_estimates.append(coeffs[0] * np.tan(x) + coeffs[1])
    return squared_error(y_estimates, y_test)


def find_cos_error(coeffs, x_test, y_test):
    y_estimates = []
    for x in x_test:
        y_estimates.append(coeffs[0] * np.cos(x) + coeffs[1])
    return squared_error(y_estimates, y_test)


def find_sin_error(coeffs, x_test, y_test):
    y_estimates = []
    for x in x_test:
        y_estimates.append(coeffs[0] * np.sin(x) + coeffs[1])
    return squared_error(y_estimates, y_test)


def find_best(all_results):
    i = 0
    # I tried using sorted but it didn't seem to work
    smallest_error_so_far = 99999999999  # set to really high, so that it is overwritten
    best_result = []
    for result in all_results:
        if result[1] < smallest_error_so_far:
            smallest_error_so_far = result[1]
            best_result = result
        i += 1
    return best_result


def cross_validation():
    j = 0
    all_results = []
    # Using 80% of the data for training
    while j in range(1):  # repeating 100 times to get good results
        x_train, y_train = np.empty(16), np.empty(16)
        x_test, y_test = np.empty(4), np.empty(4)
        options = [i for i in range(0, 20)]  # using X for no reason, could be Y as len(X) == len(Y)
        for i in range(4):  # it won't necessarily add a new one every time
            # randomly choosing the 4 elements of X and Y for testing
            for_testing = random.choice(options)
            x_test[i] = X[for_testing]
            y_test[i] = Y[for_testing]
            options.remove(for_testing)
        # the remaining options are for training
        for i in range(16):
            x_train[i] = X[options[i]]
            y_train[i] = Y[options[i]]

        sin_xs, sin_ys, sin_coefs = sin_line(x_train, y_train)
        cos_xs, cos_ys, cos_coefs = cos_line(x_train, y_train)
        tan_xs, tan_ys, tan_coefs = tan_line(x_train, y_train)
        exp_xs, exp_ys, exp_coefs = exp_line(x_train, y_train)

        # calculating errors
        sin_error = find_sin_error(sin_coefs, x_test, y_test)
        cos_error = find_cos_error(cos_coefs, x_test, y_test)
        tan_error = find_tan_error(tan_coefs, x_test, y_test)
        exp_error = find_exp_error(exp_coefs, x_test, y_test)

        # deciding which is better, and storing it
        this_error = min(sin_error, cos_error, tan_error, exp_error)
        if this_error == sin_error:
            all_results.append(["sin", this_error])
        elif this_error == cos_error:
            all_results.append(["cos", this_error])
        elif this_error == tan_error:
            all_results.append(["tan", this_error])
        elif this_error == exp_error:
            all_results.append(["exp", this_error])
        j += 1
    return all_results


def run_calculations():
    chosen_one = find_best(cross_validation())
    # recalculating the lines and errors for the whole data set and returning it
    print("Chose ", chosen_one[0], " with error of ", chosen_one[1])
    true_xs, true_ys, true_error = 0, 0, chosen_one[1]

    # if chosen_one[0] == "sin":
    #     true_xs, true_ys, true_coefficient = sin_line(X, Y)
    #     true_error = find_sin_error(true_coefficient, X, Y)
    # elif chosen_one[0] == "cos":
    #     true_xs, true_ys, true_coefficient = cos_line(X, Y)
    #     true_error = find_cos_error(true_coefficient, X, Y)
    # elif chosen_one[0] == "tan":
    #     true_xs, true_ys, true_coefficient = tan_line(X, Y)
    #     true_error = find_tan_error(true_coefficient, X, Y)
    # elif chosen_one[0] == "exp":
    #     true_xs, true_ys, true_coefficient = exp_line(X, Y)
    #     true_error = find_exp_error(true_coefficient, X, Y)
    return true_xs, true_ys, true_error


tableau20 = [(31, 119, 180), (174, 199, 232), (255, 127, 14), (255, 187, 120),
             (44, 160, 44), (152, 223, 138), (214, 39, 40), (255, 152, 150),
             (148, 103, 189), (197, 176, 213), (140, 86, 75), (196, 156, 148),
             (227, 119, 194), (247, 182, 210), (127, 127, 127), (199, 199, 199),
             (188, 189, 34), (219, 219, 141), (23, 190, 207), (158, 218, 229)]

# Scale the RGB values to the [0, 1] range, which is the format matplotlib accepts.
for i in range(len(tableau20)):
    r, g, b = tableau20[i]
    tableau20[i] = (r / 255., g / 255., b / 255.)

datafile = sys.argv[1]  # sys.argv contains the arguments passed to the program
totalX, totalY = load_points_from_file(datafile)
lines = []
start = 0
end = 20
error = 0
while end <= len(totalX):  # data is in chunks of 20
    X = totalX[start:end]
    Y = totalY[start:end]
    line_xs, line_ys, calculated_error = run_calculations()
    error += calculated_error
    # plt.legend(lines, ['Order 2', 'Order 3', 'Order 4', 'Order 5', 'Order 6', 'Order 7', 'Order 8', 'Order 9', 'Order 10'])
    # plt.plot(line_xs, line_ys, 'r-', lw=3)
    view_data_segments(X, Y)  # show each section
    start += 20
    end += 20
if len(sys.argv) >= 3:
    if sys.argv[2] == "--plot":
        view_data_segments(totalX, totalY)  # show each section
print(error)
