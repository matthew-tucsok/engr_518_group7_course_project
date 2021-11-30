import os
import random

from data_loader import DataLoader
from model import Model


def main():
    cwd = os.getcwd()
    os.chdir(cwd)
    training_dataset = DataLoader(cwd + '/Small Training Dataset')
    training_dataset.vectorize_data()
    classifier = Model(training_dataset.samples, 0.001, 1e-7)

    test_dataset = DataLoader(cwd + '/Small Test Dataset')
    test_dataset.vectorize_data()
    confusion_matrix = classifier.accuracy(test_dataset.samples)
    print(confusion_matrix)


if __name__ == '__main__':
    main()
