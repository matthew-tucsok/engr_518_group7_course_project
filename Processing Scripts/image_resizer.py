"""
ENGR 518
Authors: Adapted from Aliakbar Davoodi and written by Matthew Tucsok
Name: Image Preprocessing Script
Function: Converts arbitrary rgb .png images to 64x64 grayscale .png images
"""

from PIL import Image, ImageOps

import os
import glob

"""
How to Use this Script:
0. Be sure to install Pillow to your virtual environment, as it is not a standard library package.
1. Put only either non-flat, or flat images into the "Raw Images" folder. Do not put a mix of the two into this folder.
Flat and non-flat images must be processed separately to ensure the naming convention.
2. Change the variable "image_class" to either "nf_" or "f_" corresponding to what type of image set you are trying to 
process
3. Change the variable "image_id" to correspond to the starting id for you images
4. Run the script and the images should be renamed and resized/greyscaled. These images are saved to the folder 
"Greyscale Images"
"""


def main():
    resize_dim = 64
    image_class = 'nf_'
    image_id = 3500  # Start at this image ID, and increment sequentially

    cwd = os.getcwd()  # Getting current working directory
    os.chdir(cwd)  # Changing to current working directory
    # This will recursively find all files within the Raw images folder and find the .png images
    files = glob.glob(cwd + '/Raw Images/*.png', recursive=True)
    for file in files:
        img = Image.open(file)
        # The image will be cropped to a square aspect ration before resizing to avoid skewing
        width, height = img.size
        # Determining if the width or the height is smaller in order to crop by the smallest dimension
        if width < height:
            crop_val = width
        else:
            crop_val = height
        img = img.crop((0, 0, crop_val, crop_val))  # Cropping
        img = img.resize((resize_dim, resize_dim))  # Resizing
        img = ImageOps.grayscale(img)  # Converting to grayscale
        # Saving the image with the desired class and id to disk
        img.save(cwd + '/Greyscale Images/' + image_class + str(image_id) + '.png')
        image_id += 1


if __name__ == '__main__':
    main()
