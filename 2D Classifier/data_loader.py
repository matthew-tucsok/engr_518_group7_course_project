import glob

from PIL import Image, ImageFilter
import numpy as np


class DataLoader:
    """
    This class manages how images are loaded in, preprocessed, and assembled into an input format compatible with our
    2D classifier model. Note that due to how we create our training and test sets, both sets get preprocessed.
    """
    def __init__(self, path):
        """
        Initializing the data loader to find the dataset images.
        :param path: Path to the root of the program. The root should contain the executing script, and a folder
        named "Greyscale Dataset" which directly has the 4000 images in the dataset.
        """
        self.source = path
        self.samples = []  # List to hold the dataset samples
        self.calculated_features = []  # List to hold the calculated features that will be used as input to the model
        self.load_data()  # Begin data loading method

    def load_data(self):
        files = glob.glob(self.source + '/*.png')  # This will put all the .png filenames into a list

        # index = 0
        for file in files:  # Iterate over all files in the dataset
            img = Image.open(file).convert('L')  # Open the greyscale images, specifying 'L' to ensure correct loading

            """
            The following 3 line were for visualization purposes only in the report. They select a specific image to
            show how the original greyscale image looks before pre-processing. 
            """
            # grey = Image.open(file).convert('L')
            # if index % 3334 == 0 and index > 0:
            #     img.show()

            """
            # img_h is a histogram with 256 bins with each bin n counting how many times a pixel with a value of n
            appears in the image. 
            """
            img_h = np.array(img.histogram())
            var = np.var(img_h)  # Calculating the variance of the histogram. This is one of our final features

            """
            img.filter(ImageFilter.FIND_EDGES) generates an edge-detected image with the same dimensions as the original
            greyscale image. 
            """
            img = img.filter(ImageFilter.FIND_EDGES)

            # The following 2 lines were for showing the edge-detected image for the report
            # if index % 3334 == 0 and index > 0:
            #     img.show()

            img_array = np.array(img)  # Ensuring PIL image is now a numpy array

            # Calculating the mean of the edge-detected image
            mean = np.sum(img_array)/(img_array.shape[0]*img_array.shape[1])

            binary_img = np.where(img_array > mean, 1, 0)  # Thresholding edge-detected image based on mean pixel value
            edge_pixel_sum = np.sum(binary_img)  # Summing all of the pixel values in the binary image. 2nd final feature

            # Following 4 lines used to display the binary image for visualization in the report.
            # if index % 3334 == 0 and index > 0:
            #     print(index)
            #     pil_binary_img = Image.fromarray(binary_img*255)
            #     pil_binary_img.show()

            splits = file.split('\\')  # Parsing the file name for label extraction

            class_and_index = splits[-1]  # Getting the last split which contain the name of the image with filetype
            label_name, _ = class_and_index.split('_', 2)  # Getting only the 'f' or 'nf' part of the image name
            label = None
            if label_name == 'f':
                label = 1  # Flat images are class 1
            elif label_name == 'nf':
                label = 0  # Non-flat images are class 0
            else:
                raise SyntaxError('Invalid class label detected')  # Just a check to see if data was loaded correctly
            self.calculated_features.append(np.array([var, edge_pixel_sum]))  # Appending the features to the feature list
            # self.samples.append([img, label])  # (Deprecated)
            self.samples.append([binary_img, label])
            # self.samples.append([grey, label])
            # index += 1

        """
        The following block of code is where the final feature vector is generated as input to the model. This section
        contains commented code and visibly redundant lines to keep above code consistent for all iterations of input
        while still maintaining functionality. For example, in the final feature vector we do not include image pixels,
        so vectorization of the image is no longer required and therefore replaced with an empty array to remove these
        features
        """
        index = 0
        for sample in self.samples:  # Iterating over all samples in the dataset
            img_array = np.array(sample[0])  # (Deprecated) Extracting the previous image associated with a sample
            img_vector = img_array.ravel()  # (Deprecated) vectorizing the 64 x 64 image to a 4096 x 1 vector
            img_vector = np.array([])  # Removing the image vector as it was not used in the final iteration of the model

            # (Deprecated) Used to remove calculated features to test only the performance of vectorized image inputs
            # self.calculated_features[index] = np.array([])

            # The line below will keep whatever features still exist after the decisions made above
            sample_vector = np.hstack((img_vector, self.calculated_features[index]))
            self.samples[index][0] = sample_vector
            index += 1
