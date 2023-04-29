import os
import numpy as np
from PIL import Image, ImageOps
from numpy import asarray


def main(path):
    img = Image.open(path)
    img = ImageOps.grayscale(img)
    numpydata = asarray(img)
    filename, file_extension = os.path.splitext(path)
    np.save(filename + "_arr", numpydata)


if __name__ == "__main__":
    dir_path = os.path.dirname(os.path.realpath(
        __file__))  # Get current directory path
    dirs = os.listdir(dir_path)  # Get all files in current directory
    for file in dirs:
        if file == "cabbage" or file == "carrot" or file == "tomato":
            data_file = os.path.join(dir_path, file)
            for img in os.listdir(data_file):
                img_path = os.path.join(data_file, img)
                print("Converting " + img_path + " to numpy array")
                main(img_path)
