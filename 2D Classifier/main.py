import os
import random
import time

from data_loader import DataLoader
from model import Model

import numpy as np


def main():
    """
    The main executing function. This function is responsible for finding the path to our dataset, creating an instance
    of our dataset, creating an instance of our model, and running the cross-validation of our model.
    """
    cwd = os.getcwd()  # Getting the root directory of the folder that executed the main function
    os.chdir(cwd)  # Ensuring this is also set to the current directory

    """
    # "runs" controls how many times the model is trained and tested. This should be performed multiple times for 
    evaluating model accuracy since the training and testing sets are randomly generated with a roughly 50/50 split of
    non-flat and flat images in each set. This ensemble cross-validation method of evaluation uses the law of large numbers to find the
    true general accuracy of the final model. 
    """
    runs = 100  # Large enough that average accuracy will have a decent confidence based on variance of model
    accuracies = []  # List to store test accuracies for cross validation
    start = time.time()  # Start time for timing the total execution of the multiple runs of the program
    for run in range(runs):  # Each run is an independent evaluation of performance
        print('Run', run + 1)
        dataset = DataLoader(cwd + '/Greyscale Dataset')  # Creating an instance of the dataset
        random.shuffle(dataset.samples)  # Randomizing the order of the dataset to ensure order invariance
        train_test_split = 0.75  # We selected a 75% training 25% testing for the model training and evaluation
        split_index = round(len(dataset.samples) * train_test_split)  # Determining where to split the dataset
        training_set = dataset.samples[:split_index]  # Extracting the training set from the shuffled dataset
        test_set = dataset.samples[split_index:]  # Extracting the testing set from the shuffled dataset

        """
        The following block of code is for visualizing how many images are flat and how many are non-flat for a training
        set. This was to see if the model is biased towards the class with more examples. This did not seem to be too
        much of a problem in testing.
        """
        flat_label_count = 0
        not_flat_label_count = 0
        for sample in training_set:
            if sample[1] == 1:
                flat_label_count += 1
            else:
                not_flat_label_count += 1
        print(flat_label_count, 'flat images and', not_flat_label_count, 'non-flat images in training set.')

        classifier = Model(training_set)  # Creating an instance of the classifier. This initiates model training.

        """
        Code block below checks the number of flat and non-flat images in the test set.
        """
        flat_label_count = 0
        not_flat_label_count = 0
        for sample in test_set:
            if sample[1] == 1:
                flat_label_count += 1
            else:
                not_flat_label_count += 1
        print(flat_label_count, 'flat images and', not_flat_label_count, 'non-flat images in test set.')

        confusion_matrix = classifier.accuracy(training_set)  # Getting the confusion matrix of the training_set
        accuracy = 100 * np.trace(confusion_matrix) / np.sum(confusion_matrix)  # Calculating accuracy of training set
        print(confusion_matrix)
        print('Training Accuracy:', accuracy, '%')

        confusion_matrix = classifier.accuracy(test_set)  # Getting the confusion matrix of the test set
        print(confusion_matrix)
        accuracy = 100 * np.trace(confusion_matrix) / np.sum(confusion_matrix)  # Calculating accuracy of test set
        accuracies.append(accuracy)  # Appending only the test accuracy to the list of accuracy for cross-validation
        print('Testing Accuracy:', accuracy, '%')

    accuracies = np.array(accuracies)  # Converting python list to numpy array for accuracies
    mean_accuracy = np.mean(accuracies)
    var_accuracy = np.var(accuracies)  # Variance of accuracies
    std_accuracy = np.sqrt(var_accuracy)  # Standard deviation of accuracies
    print('Average of Accuracy:', mean_accuracy, '%')
    print('Variance of Accuracy:', var_accuracy)
    print('Standard Deviation of Accuracy:', std_accuracy)
    print('Runtime for', runs, 'runs:', round(time.time() - start, 2), 'seconds')  # Complete runtime for the runs


if __name__ == '__main__':  # Python convention for ensuring recursive calling of main function does not occur
    main()
