import os
import random

from data_loader import DataLoader
from model import Model

import numpy as np


def main():
    cwd = os.getcwd()
    os.chdir(cwd)

    runs = 10
    accuracies = []
    for run in range(runs):
        print('Run', run+1)
        dataset = DataLoader(cwd + '/Greyscale Dataset')
        train_test_split = 0.75

        random.shuffle(dataset.samples)
        train_test_split = 0.75
        split_index = round(len(dataset.samples) * train_test_split)
        training_set = dataset.samples[:split_index]
        test_set = dataset.samples[split_index:]

        flat_label_count = 0
        not_flat_label_count = 0
        for sample in training_set:
            if sample[1] == 1:
                flat_label_count += 1
            else:
                not_flat_label_count += 1

        print(flat_label_count, 'flat images and', not_flat_label_count, 'non-flat images in training set.')

        classifier = Model(training_set, 0.001, 1e-7)

        flat_label_count = 0
        not_flat_label_count = 0
        for sample in test_set:
            if sample[1] == 1:
                flat_label_count += 1
            else:
                not_flat_label_count += 1

        print(flat_label_count, 'flat images and', not_flat_label_count, 'non-flat images in test set.')

        confusion_matrix = classifier.accuracy(test_set)
        print(confusion_matrix)
        accuracy = 100 * np.trace(confusion_matrix) / np.sum(confusion_matrix)
        accuracies.append(accuracy)
        print('Accuracy:', accuracy, '%')

    accuracies = np.array(accuracies)
    mean_accuracy = np.mean(accuracies)
    var_accuracy = np.var(accuracies)
    std_accuracy = np.sqrt(var_accuracy)
    print('Average of Accuracy:', mean_accuracy, '%')
    print('Variance of Accuracy:', var_accuracy)
    print('Standard Deviation of Accuracy:', std_accuracy)


if __name__ == '__main__':
    main()
