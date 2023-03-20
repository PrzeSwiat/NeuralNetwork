import csv
import random
import time
from matplotlib import pyplot as plt
import numpy as np

from matrix import fromArray
from neuralNetwork import NeuralNetwork

auto_encoder = [[[1, 0, 0, 0], [1, 0, 0, 0]], [[0, 1, 0, 0], [0, 1, 0, 0]], [[0, 0, 1, 0], [0, 0, 1, 0]],
                [[0, 0, 0, 1], [0, 0, 0, 1]]]


def setup():
    nn = NeuralNetwork(4, 1, 2, 4, 1)
    print("Choose database to train:")
    print("1 - Iris dataset")
    print("2 - Autoencoder")
    choose = int(input())
    dataset = 0
    if choose == 1:
        dataset = iris_dataset()
    if choose == 2:
        dataset = auto_encoder
    print("Do You want to shuffle training pattern: ")
    print("y/n [Press Enter to confirm]")
    stop = input()
    print("Choose the number of epochs of training:")
    print("[Press Enter to confirm]")
    stop2 = input()
    print("Choose the error the training should stop:")
    print("[Press Enter to confirm]")
    stop3 = input()

    iterator = 0
    iterator_2 = 0
    r = 0
    error = 10000
    error_container = []
    epoch_counter = []
    epoch_error = []
    one = 0
    t0 = time.process_time()
    for n in range(int(stop2)):
        if error >= float(stop3):
            if stop == "y":
                if choose == 1:
                    r = random.randint(0, 8)
                if choose == 2:
                    r = random.randint(0, 3)
            if stop == "n":
                r = iterator
            if choose == 1:
                error = nn.train(dataset[r][0][0], dataset[r][1][0])
            if choose == 2:
                error = nn.train(dataset[r][0], dataset[r][1])
            epoch_error.append(error)
            one += 1
            if iterator_2 == 4:
                epoch_counter.append(one)
                global_error = 0
                for each in epoch_error:
                    global_error += each
                global_error = global_error/4
                error_container.append(global_error)
                epoch_error.clear()
                iterator_2 = 0
            iterator += 1
            iterator_2 += 1
            if iterator == 3:
                iterator = 0

    plt.plot(epoch_counter, error_container)
    plt.title("Error of network")
    plt.show()
    # nn.write_network_to_file("test")
    # nn.read_network_from_file("test")
    # AUTO ENCODER CHECKS
    # ///////////////////
    nn.feedforward_result([1, 0, 0, 0], [1, 0, 0, 0])
    nn.feedforward_result([0, 1, 0, 0], [0, 1, 0, 0])
    nn.feedforward_result([0, 0, 1, 0], [0, 0, 1, 0])
    nn.feedforward_result([0, 0, 0, 1], [0, 0, 0, 1])
    # ///////////////////

    # IRIS DATASET CHECKS
    # iris_data_set_checks(nn)
    # nn.feedforward_result([4.9, 3.0, 1.4, 0.2], [1, 0, 0])
    # nn.feedforward_result([6.4, 3.2, 4.5, 1.5], [0, 1, 0])
    # nn.feedforward_result([7.1, 3.0, 5.9, 2.1], [0, 0, 1])
    # ///////////////////

    print()
    print("Time of training process: " + str(time.process_time() - t0))


def iris_dataset():
    # for training
    file = open("data.csv", newline='')
    reader = csv.reader(file)
    iris_set = []
    training_set = []
    accurate = []
    database = []
    iterator = 0
    for row in reader:
        iris_set.append(row)

    splits = np.array_split(iris_set, 50)
    rand_1 = 0
    rand_2 = 17
    rand_3 = 34
    for one in splits:
        if iterator == rand_1 or iterator == rand_2 or iterator == rand_3:
            training_set.append(one)
        iterator += 1
    for one in training_set:
        for each in one:
            accurate.append(each)
    for row in accurate:
        inputs = []
        outputs = []
        inputs.append([float(row[0]), float(row[1]), float(row[2]), float(row[3])])
        new = []
        if float(row[4]) == 0:
            new = [1, 0, 0]
        if float(row[4]) == 1:
            new = [0, 1, 0]
        if float(row[4]) == 2:
            new = [0, 0, 1]
        outputs.append(new)
        database.append([inputs, outputs])
    return database


def iris_data_set_checks(nn):
    file = open("data.csv", newline='')
    reader = csv.reader(file)
    database = []
    for row in reader:
        inputs = []
        outputs = []
        inputs.append([float(row[0]), float(row[1]), float(row[2]), float(row[3])])
        new = []
        if float(row[4]) == 0:
            new = [1, 0, 0]
        if float(row[4]) == 1:
            new = [0, 1, 0]
        if float(row[4]) == 2:
            new = [0, 0, 1]
        outputs.append(new)
        database.append([inputs, outputs])

    # Classification staff:
    # ///////////////////////
    first = 0
    two = 0
    three = 0
    factor = 0
    row_iterator = 0
    one_row_class = []
    error_container = []
    for one in database:
        new = nn.feedforward(one[0][0])
        # Computing error of network
        # //////////////
        target = one[1][0]
        target = fromArray(target)
        whole_error = target.subtract(new[0])
        sum_error = 0
        for i in range(whole_error.rows):
            for j in range(whole_error.columns):
                sum_error += abs(whole_error.matrix[i][j])
        sum_error = sum_error / whole_error.rows
        error_container.append(sum_error)

        # sum of error below...
        # //////////////
        output = new[0].toArray()

        iterator = 0
        for out in output:
            if out >= 0.5:
                guess = 1
            else:
                guess = 0
            if iterator == 0 and guess == 1:
                first += 1
            if iterator == 1 and guess == 1:
                two += 1
            if iterator == 2 and guess == 1:
                three += 1
            iterator += 1
        row_iterator += 1
        if row_iterator % 50 == 0:
            one_row_class.append([first, two, three])
            first = 0
            two = 0
            three = 0
            factor += 1
    sum_network_error = 0
    for each in error_container:
        sum_network_error += each
    sum_network_error = sum_network_error / 150
    print("Arithmetic sum of error: " + str(round(sum_network_error, 6)))
    # //////////////
    # CONFUSION MATRIX
    s = (3, 3)
    confusion_matrix = np.zeros(s, dtype=int)
    iterator_i = 0
    for each in one_row_class:
        iterator_j = 0
        for one in each:
            confusion_matrix[iterator_j][iterator_i] = one
            iterator_j += 1
        iterator_i += 1

    print("Confusion matrix:")
    print(confusion_matrix)
    # //////////////

    # PRECISION
    # //////////////
    setosa_prec = round(
        confusion_matrix[0][0] / (confusion_matrix[0][0] + confusion_matrix[0][1] + confusion_matrix[0][2]), 2)
    versi_prec = round(
        confusion_matrix[1][1] / (confusion_matrix[1][0] + confusion_matrix[1][1] + confusion_matrix[1][2]), 2)
    virgi_prec = round(
        confusion_matrix[2][2] / (confusion_matrix[2][0] + confusion_matrix[2][1] + confusion_matrix[2][2]), 2)
    print("Precision for the Setosa class: " + str(setosa_prec * 100) + "%")
    print("Precision for the Versicolor class: " + str(versi_prec * 100) + "%")
    print("Precision for the Virginica class: " + str(virgi_prec * 100) + "%")
    # //////////////
    print()
    # RECALL
    # //////////////
    setosa_reca = round(
        confusion_matrix[0][0] / (confusion_matrix[0][0] + confusion_matrix[1][0] + confusion_matrix[2][0]), 2)
    versi_reca = round(
        confusion_matrix[1][1] / (confusion_matrix[0][1] + confusion_matrix[1][1] + confusion_matrix[2][1]), 2)
    virgi_reca = round(
        confusion_matrix[2][2] / (confusion_matrix[0][2] + confusion_matrix[1][2] + confusion_matrix[2][2]), 2)
    print("Recall for the Setosa class: " + str(setosa_reca * 100) + "%")
    print("Recall for the Versicolor class: " + str(versi_reca * 100) + "%")
    print("Recall for the Virginica class: " + str(virgi_reca * 100) + "%")
    # //////////////
    print()
    # F MEASURE
    # //////////////
    f1_setosa = round((2 * (setosa_reca * setosa_prec)) / (setosa_reca + setosa_prec), 2)
    f1_versi = round((2 * (versi_reca * versi_prec)) / (versi_reca + versi_prec), 2)
    f1_virgi = round((2 * (virgi_reca * virgi_prec)) / (virgi_reca + virgi_prec), 2)
    print("F-Measure for the Setosa class: " + str(f1_setosa * 100) + "%")
    print("F-Measure for the Versicolor class: " + str(f1_versi * 100) + "%")
    print("F-Measure for the Virginica class: " + str(f1_virgi * 100) + "%")
    # //////////////
    # ///////////////////////


setup()
# iris_database()
