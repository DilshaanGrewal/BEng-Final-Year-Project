
import numpy as np
from PIL import Image, ImageOps
from numpy import asarray
import sys

def main():
    img = Image.open(sys.argv[1])
    img = ImageOps.grayscale(img)
    numpydata = asarray(img)
    a = sys.argv[1].split("/")
    b = a[-1].split(".")
    np.save(b[0] + "_arr", numpydata)

if __name__ == "__main__":
    main()
