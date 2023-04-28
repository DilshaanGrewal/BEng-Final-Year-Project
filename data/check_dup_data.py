import os
import numpy as np
from PIL import Image


def check_duplicates(path):
    """Check if there are duplicate images in a folder"""
    for img_file in os.listdir(path):  # Iterate through all images in folder
        img_path = os.path.join(path, img_file)
        image = Image.open(str(img_path))
        arr = np.asarray(image)  # Convert image to numpy array

        for img2_file in os.listdir(path):
            if img_file != img2_file:   # Don't compare to itself
                img2_path = os.path.join(path, img2_file)
                image2 = Image.open(str(img2_path))
                arr2 = np.asarray(image2)

                if np.array_equal(arr, arr2):
                    print("Duplicate images: " + img_file, img2_file)


if __name__ == "__main__":
    dir_path = os.path.dirname(os.path.realpath(__file__))  # Get current directory path
    dirs = os.listdir(dir_path)  # Get all files in current directory
    for file in dirs:
        data_file = os.path.join(dir_path, file)
        if os.path.isdir(data_file):
            print("Checking duplicates in " + data_file)
            check_duplicates(data_file)
