import os
import random

from data_loader import DataLoader
from model import Model



def main():
    cwd = os.getcwd()
    os.chdir(cwd)
    dataset = DataLoader(cwd + '/Greyscale Dataset')
    dataset.vectorize_data()
    random.shuffle(dataset.samples)
    train_test_split = 0.75
    split_index = round(len(dataset.samples)*train_test_split)
    training_set = dataset.samples[:split_index]
    test_set = dataset.samples[split_index:]

    classifier = Model(training_set, 0.001, 1e-7)

    confusion_matrix = classifier.accuracy(test_set)
    print(confusion_matrix)


if __name__ == '__main__':
    main()
