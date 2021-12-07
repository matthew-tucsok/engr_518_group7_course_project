import glob

from PIL import Image, ImageFilter
import numpy as np


class DataLoader:
    def __init__(self, path):
        self.source = path
        self.samples = []
        self.calculated_features = []
        self.load_data()

    def load_data(self):
        files = glob.glob(self.source + '/*.png')
        for file in files:
            img = Image.open(file).convert('L')
            img = img.filter(ImageFilter.FIND_EDGES)
            img_h = np.array(img.histogram())
            var = np.var(img_h)
            mean = np.mean(img)
            splits = file.split('\\')

            class_and_index = splits[-1]
            label_name, _ = class_and_index.split('_', 2)
            label = None
            if label_name == 'f':
                label = 1
            elif label_name == 'nf':
                label = 0
            else:
                raise SyntaxError('Invalid class label detected')
            self.calculated_features.append(np.array([var, mean]))
            # self.samples.append([img, label])
            self.samples.append([img, label])

        index = 0
        for sample in self.samples:
            img_array = np.array(sample[0])
            img_vector = img_array.ravel()
            sample_vector = np.hstack((img_vector, self.calculated_features[index]))
            self.samples[index][0] = sample_vector
            index += 1

