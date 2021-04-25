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


def least_squares(x, y):
    ones = np.ones(x.shape)
    x_with_ones = np.column_stack((ones, x))
    return least_squares_formula(x_with_ones, y)


def squared_resizer(x):
    return np.column_stack((np.ones(x.shape), x, x ** 2))


def cubed_resizer(x):
    return np.column_stack((np.ones(x.shape), x, x ** 2, x ** 3))


def four_resizer(x):
    return np.column_stack((np.ones(x.shape), x, x ** 2, x ** 3, x ** 4))


def five_resizer(x):
    return np.column_stack((np.ones(x.shape), x, x ** 2, x ** 3, x ** 4, x ** 5))


def six_resizer(x):
    return np.column_stack((np.ones(x.shape), x, x ** 2, x ** 3, x ** 4, x ** 5, x ** 6))


def seven_resizer(x):
    return np.column_stack((np.ones(x.shape), x, x ** 2, x ** 3, x ** 4, x ** 5, x ** 6, x ** 7))


def eight_resizer(x):
    return np.column_stack((np.ones(x.shape), x, x ** 2, x ** 3, x ** 4, x ** 5, x ** 6, x ** 7, x ** 8))


def nine_resizer(x):
    return np.column_stack((np.ones(x.shape), x, x ** 2, x ** 3, x ** 4, x ** 5, x ** 6, x ** 7, x ** 8, x ** 9))


def ten_resizer(x):
    return np.column_stack((np.ones(x.shape), x, x ** 2, x ** 3, x ** 4, x ** 5, x ** 6, x ** 7, x ** 8, x ** 9, x ** 10))


def squared_line(xs, ys):
    resized_x = squared_resizer(xs)
    matrix = least_squares_formula(resized_x, ys)
    resulting_x = np.linspace(X[0], X[19], 20)  # uses X over 20 so that it covers all the data
    resulting_y = squared_resizer(resulting_x) @ matrix  # @ is matrix multiplication
    plt.plot(resulting_x, resulting_y, c=tableau20[0], lw=1)
    return resulting_x, resulting_y, matrix


def cubed_line(xs, ys):
    resized_x = cubed_resizer(xs)
    matrix = least_squares_formula(resized_x, ys)
    resulting_x = np.linspace(X[0], X[19], 20)  # uses X over 20 so that it covers all the data
    resulting_y = cubed_resizer(resulting_x) @ matrix  # @ is matrix multiplication
    plt.plot(resulting_x, resulting_y, c=tableau20[2], lw=1)
    return resulting_x, resulting_y, matrix


def four_line(xs, ys):
    resized_x = four_resizer(xs)
    matrix = least_squares_formula(resized_x, ys)
    resulting_x = np.linspace(X[0], X[19], 20)  # uses X over 20 so that it covers all the data
    resulting_y = four_resizer(resulting_x) @ matrix  # @ is matrix multiplication
    plt.plot(resulting_x, resulting_y, c=tableau20[4], lw=1)
    return resulting_x, resulting_y, matrix


def five_line(xs, ys):
    resized_x = five_resizer(xs)
    matrix = least_squares_formula(resized_x, ys)
    resulting_x = np.linspace(X[0], X[19], 20)  # uses X over 20 so that it covers all the data
    resulting_y = five_resizer(resulting_x) @ matrix  # @ is matrix multiplication
    plt.plot(resulting_x, resulting_y, c=tableau20[6], lw=1)
    return resulting_x, resulting_y, matrix


def six_line(xs, ys):
    resized_x = six_resizer(xs)
    matrix = least_squares_formula(resized_x, ys)
    resulting_x = np.linspace(X[0], X[19], 20)  # uses X over 20 so that it covers all the data
    resulting_y = six_resizer(resulting_x) @ matrix  # @ is matrix multiplication
    plt.plot(resulting_x, resulting_y, c=tableau20[8], lw=1)
    return resulting_x, resulting_y, matrix


def seven_line(xs, ys):
    resized_x = seven_resizer(xs)
    matrix = least_squares_formula(resized_x, ys)
    resulting_x = np.linspace(X[0], X[19], 20)  # uses X over 20 so that it covers all the data
    resulting_y = seven_resizer(resulting_x) @ matrix  # @ is matrix multiplication
    plt.plot(resulting_x, resulting_y, c=tableau20[10], lw=1)
    return resulting_x, resulting_y, matrix


def eight_line(xs, ys):
    resized_x = eight_resizer(xs)
    matrix = least_squares_formula(resized_x, ys)
    resulting_x = np.linspace(X[0], X[19], 20)  # uses X over 20 so that it covers all the data
    resulting_y = eight_resizer(resulting_x) @ matrix  # @ is matrix multiplication
    plt.plot(resulting_x, resulting_y, c=tableau20[12], lw=1)
    return resulting_x, resulting_y, matrix


def nine_line(xs, ys):
    resized_x = nine_resizer(xs)
    matrix = least_squares_formula(resized_x, ys)
    resulting_x = np.linspace(X[0], X[19], 20)  # uses X over 20 so that it covers all the data
    resulting_y = nine_resizer(resulting_x) @ matrix  # @ is matrix multiplication
    plt.plot(resulting_x, resulting_y, c=tableau20[15], lw=1)
    return resulting_x, resulting_y, matrix


def ten_line(xs, ys):
    resized_x = ten_resizer(xs)
    matrix = least_squares_formula(resized_x, ys)
    resulting_x = np.linspace(X[0], X[19], 20)  # uses X over 20 so that it covers all the data
    resulting_y = ten_resizer(resulting_x) @ matrix  # @ is matrix multiplication
    plt.plot(resulting_x, resulting_y, c=tableau20[19], lw=1)
    return resulting_x, resulting_y, matrix


def squared_error(estimated_y, test_y):
    this_error = 0
    for i in range(len(estimated_y)):
        squared_difference = (estimated_y[i] - test_y[i]) ** 2
        this_error += squared_difference
    return this_error


def find_squared_error(coeffs, x_test, y_test):
    y_estimates = []
    for x in x_test:
        y_estimates.append((coeffs[2] * (x ** 2)) + (coeffs[1] * x) + coeffs[0])
    return squared_error(y_estimates, y_test)


def find_cubed_error(coeffs, x_test, y_test):
    y_estimates = []
    for x in x_test:
        y_estimates.append((coeffs[3] * (x ** 3)) + (coeffs[2] * (x ** 2)) + (coeffs[1] * x) + coeffs[0])
    return squared_error(y_estimates, y_test)


def find_four_error(coeffs, x_test, y_test):
    y_estimates = []
    for x in x_test:
        y_estimates.append((coeffs[4] * (x ** 4)) + (coeffs[3] * (x ** 3)) + (coeffs[2] * (x ** 2)) + (coeffs[1] * x) + coeffs[0])
    return squared_error(y_estimates, y_test)


def find_five_error(coeffs, x_test, y_test):
    y_estimates = []
    for x in x_test:
        y_estimates.append((coeffs[5] * (x ** 5)) + (coeffs[4] * (x ** 4)) + (coeffs[3] * (x ** 3)) + (coeffs[2] * (x ** 2)) + (coeffs[1] * x) + coeffs[0])
    return squared_error(y_estimates, y_test)


def find_six_error(coeffs, x_test, y_test):
    y_estimates = []
    for x in x_test:
        y_estimates.append((coeffs[6] * (x ** 6)) + (coeffs[5] * (x ** 5)) + (coeffs[4] * (x ** 4)) + (coeffs[3] * (x ** 3)) + (coeffs[2] * (x ** 2)) + (coeffs[1] * x) + coeffs[0])
    return squared_error(y_estimates, y_test)


def find_seven_error(coeffs, x_test, y_test):
    y_estimates = []
    for x in x_test:
        y_estimates.append((coeffs[7] * (x ** 7)) + (coeffs[6] * (x ** 6)) + (coeffs[5] * (x ** 5)) + (coeffs[4] * (x ** 4)) + (coeffs[3] * (x ** 3)) + (coeffs[2] * (x ** 2)) + (coeffs[1] * x) + coeffs[0])
    return squared_error(y_estimates, y_test)


def find_eight_error(coeffs, x_test, y_test):
    y_estimates = []
    for x in x_test:
        y_estimates.append((coeffs[8] * (x ** 8)) + (coeffs[7] * (x ** 7)) + (coeffs[6] * (x ** 6)) + (coeffs[5] * (x ** 5)) + (coeffs[4] * (x ** 4)) + (coeffs[3] * (x ** 3)) + (coeffs[2] * (x ** 2)) + (coeffs[1] * x) + coeffs[0])
    return squared_error(y_estimates, y_test)


def find_nine_error(coeffs, x_test, y_test):
    y_estimates = []
    for x in x_test:
        y_estimates.append((coeffs[9] * (x ** 9)) + (coeffs[8] * (x ** 8)) + (coeffs[7] * (x ** 7)) + (coeffs[6] * (x ** 6)) + (coeffs[5] * (x ** 5)) + (coeffs[4] * (x ** 4)) + (coeffs[3] * (x ** 3)) + (coeffs[2] * (x ** 2)) + (coeffs[1] * x) + coeffs[0])
    return squared_error(y_estimates, y_test)


def find_ten_error(coeffs, x_test, y_test):
    y_estimates = []
    for x in x_test:
        y_estimates.append((coeffs[10] * (x ** 10)) + (coeffs[9] * (x ** 9)) + (coeffs[8] * (x ** 8)) + (coeffs[7] * (x ** 7)) + (coeffs[6] * (x ** 6)) + (coeffs[5] * (x ** 5)) + (coeffs[4] * (x ** 4)) + (coeffs[3] * (x ** 3)) + (coeffs[2] * (x ** 2)) + (coeffs[1] * x) + coeffs[0])
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
    while j in range(100):  # repeating 100 times to get good results
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

        squared_xs, squared_ys, squared_coefs = squared_line(x_train, y_train)
        cubed_xs, cubed_ys, cubed_coefs = cubed_line(x_train, y_train)
        four_xs, four_ys, four_coefs = four_line(x_train, y_train)
        five_xs, five_ys, five_coefs = five_line(x_train, y_train)
        six_xs, six_ys, six_coefs = six_line(x_train, y_train)
        seven_xs, seven_ys, seven_coefs = seven_line(x_train, y_train)
        eight_xs, eight_ys, eight_coefs = eight_line(x_train, y_train)
        nine_xs, nine_ys, nine_coefs = nine_line(x_train, y_train)
        ten_xs, ten_ys, ten_coefs = ten_line(x_train, y_train)

        # calculating errors
        two_error = find_squared_error(squared_coefs, x_test, y_test)
        cubed_error = find_cubed_error(cubed_coefs, x_test, y_test)
        four_error = find_four_error(four_coefs, x_test, y_test)
        five_error = find_five_error(five_coefs, x_test, y_test)
        six_error = find_six_error(six_coefs, x_test, y_test)
        seven_error = find_seven_error(seven_coefs, x_test, y_test)
        eight_error = find_eight_error(eight_coefs, x_test, y_test)
        nine_error = find_nine_error(nine_coefs, x_test, y_test)
        ten_error = find_ten_error(ten_coefs, x_test, y_test)

        # deciding which is better, and storing it
        this_error = min(two_error, cubed_error, four_error, five_error, six_error, seven_error, eight_error, nine_error, ten_error)
        if this_error == two_error:
            all_results.append(["squared", this_error])
        elif this_error == cubed_error:
            all_results.append(["cubed", this_error])
        elif this_error == four_error:
            all_results.append(["four", this_error])
        elif this_error == five_error:
            all_results.append(["five", this_error])
        elif this_error == six_error:
            all_results.append(["six", this_error])
        elif this_error == seven_error:
            all_results.append(["seven", this_error])
        elif this_error == eight_error:
            all_results.append(["eight", this_error])
        elif this_error == nine_error:
            all_results.append(["nine", this_error])
        elif this_error == ten_error:
            all_results.append(["ten", this_error])
        j += 1
    return all_results


def run_calculations():
    chosen_one = find_best(cross_validation())
    # recalculating the lines and errors for the whole data set and returning it
    print("Chose ", chosen_one[0], " with error of ", chosen_one[1])
    if chosen_one[0] == "squared":
        true_xs, true_ys, true_coefficient = squared_line(X, Y)
        true_error = find_squared_error(true_coefficient, X, Y)
    elif chosen_one[0] == "cubed":
        true_xs, true_ys, true_coefficient = cubed_line(X, Y)
        true_error = find_cubed_error(true_coefficient, X, Y)
    elif chosen_one[0] == "four":
        true_xs, true_ys, true_coefficient = four_line(X, Y)
        true_error = find_four_error(true_coefficient, X, Y)
    elif chosen_one[0] == "five":
        true_xs, true_ys, true_coefficient = five_line(X, Y)
        true_error = find_five_error(true_coefficient, X, Y)
    elif chosen_one[0] == "six":
        true_xs, true_ys, true_coefficient = six_line(X, Y)
        true_error = find_six_error(true_coefficient, X, Y)
    elif chosen_one[0] == "seven":
        true_xs, true_ys, true_coefficient = seven_line(X, Y)
        true_error = find_seven_error(true_coefficient, X, Y)
    elif chosen_one[0] == "eight":
        true_xs, true_ys, true_coefficient = eight_line(X, Y)
        true_error = find_eight_error(true_coefficient, X, Y)
    elif chosen_one[0] == "nine":
        true_xs, true_ys, true_coefficient = nine_line(X, Y)
        true_error = find_nine_error(true_coefficient, X, Y)
    elif chosen_one[0] == "ten":
        true_xs, true_ys, true_coefficient = ten_line(X, Y)
        true_error = find_ten_error(true_coefficient, X, Y)
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
