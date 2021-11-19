import os

from data_loader import DataLoader


def main():
    cwd = os.getcwd()
    os.chdir(cwd)
    dataset = DataLoader(cwd + '/Greyscale Dataset')
    dataset.vectorize_data()
    print(len(dataset.samples[0][0]))
    print(dataset.samples[0][1])


if __name__ == '__main__':
    main()

