""""This assignment done by using Python 3.7.0"""
"""I use starter code as a guide for this assignment and implement my codes on it"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def sigmoid(x):
    return 1 / (1 + np.exp(-0.005*x))


def sigmoid_derivative(x):
    return 0.005 * x * (1 - x)

def read_and_divide_into_train_and_test(csv_file):
    data = pd.read_csv(csv_file)
    data = data[~(data["Bare_Nuclei"] == "?")].astype(int)              #Cleaning missing datas
    list = ["Code_number", "Class"]
    x = data.drop(list, axis=1)                                         #only features
    y = data.Class.values                                               #class values

    splitting_index = int(0.8 * len(x))                                 #index arrangement for splittin
    training_inputs, training_labels = np.array(x[:splitting_index]), np.array(y[:splitting_index])  #splitting data into 4, training_label&test_label&taining_input&test_input
    test_inputs, test_labels = np.array(x[splitting_index:]), np.array(y[splitting_index:])

    training_labels = training_labels.reshape(training_labels.shape[0],-1)                      #reshape datas
    test_labels = test_labels.reshape(test_labels.shape[0],-1)
    test_inputs = test_inputs.reshape(test_inputs.shape[0],-1)
    training_inputs = training_inputs.reshape(training_inputs.shape[0],-1)

    f = plt.figure(figsize=(15, 15))                                             #printing heatmap
    plt.matshow(x.corr(), fignum=f.number)
    plt.xticks(range(x.shape[1]), x.columns, fontsize=7, rotation=45)
    plt.yticks(range(x.shape[1]), x.columns, fontsize=7, rotation=0)
    cb = plt.colorbar()
    cb.ax.tick_params(labelsize=10)
    plt.show()

    return training_inputs, training_labels, test_inputs, test_labels


def run_on_test_set(test_inputs, test_labels, weights):
    tp = 0

    test_predictions = test_inputs.dot(weights)                         #Calculation of test_prediction
    test_predictions = sigmoid(test_predictions)

    test_predictions=map(lambda x: 0 if x<0.5 else 1, test_predictions) #Mapping predictions into either 0 or 1

    for predicted_val, label in zip(test_predictions, test_labels):
        if predicted_val == label:
            tp += 1

    accuracy = tp / len(test_labels)
    return accuracy


def plot_loss_accuracy(accuracy_array, loss_array):                   #function of plotting final loss,accuracy chart using matplotlib
    NumIteration = np.arange(2500)
    plt.plot(NumIteration, accuracy_array, label="Accuracy")
    plt.ylabel("Accuracy")
    plt.xlabel("Epochs")
    plt.plot(NumIteration, loss_array, label="Loss")
    plt.ylabel("Loss")
    plt.xlabel("Epochs")
    plt.legend()
    plt.show()


def main():
    csv_file = './breast-cancer-wisconsin.csv'
    iteration_count = 2500
    np.random.seed(1)
    weights = 2 * np.random.random((9, 1)) - 1
    accuracy_array = []
    loss_array = []
    training_inputs, training_labels, test_inputs, test_labels = read_and_divide_into_train_and_test(csv_file)

    for iteration in range(iteration_count):                               #forward and backward propagation here
        input = training_inputs
        output = input.dot(weights)
        output = sigmoid(output)
        loss = training_labels - output
        tuning = loss * sigmoid_derivative(output)
        weights += (input.transpose()).dot(tuning)

        loss_array.append(loss.mean())                                     #appending empty lists with datas we want to keep
        accuracy_array.append(run_on_test_set(test_inputs, test_labels, weights))

    plot_loss_accuracy(accuracy_array, loss_array)                          #Calling plotting function of accuracy and loss


if __name__ == '__main__':
    main()
