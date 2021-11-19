import os
import random

from data_loader import DataLoader
from model import Model, InClassModel


def main():
    cwd = os.getcwd()
    os.chdir(cwd)
    dataset = DataLoader(cwd + '/Greyscale Dataset')
    dataset.vectorize_data()
    # print(len(dataset.samples[0][0]))
    # print(dataset.samples[0][1])

    all_samples = dataset.samples
    random.shuffle(all_samples)
    train_number = round(len(all_samples)*0.75)
    train_samples = all_samples[0:train_number]
    test_samples = all_samples[train_number:]
    classifier = InClassModel(train_samples)
    confusion_matrix = classifier.accuracy(test_samples)
    print(confusion_matrix)

if __name__ == '__main__':
    main()
