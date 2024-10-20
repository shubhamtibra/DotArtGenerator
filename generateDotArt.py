
import cv2
from urllib.request import urlopen
import matplotlib.pyplot as plt
import numpy as np

import sys


def url_to_image(url, readFlag=cv2.IMREAD_COLOR):
    # download the image, convert it to a NumPy array, and then read
    # it into OpenCV format
    resp = urlopen(url)
    image = np.asarray(bytearray(resp.read()), dtype="uint8")
    image = cv2.imdecode(image, readFlag)

    # return the image
    return image


def resize_image_with_aspect_ratio(image, desired_width=50, aspect_ratio=None):
    # Get the original image dimensions
    height, width = image.shape[:2]

    # Calculate or use provided aspect ratio
    if aspect_ratio is None:
        aspect_ratio = height / width  # Maintain original aspect ratio

    # Calculate the new height based on desired width and aspect ratio
    new_height = int(desired_width * aspect_ratio)

    # Resize the image
    resized_img = cv2.resize(image, (desired_width, new_height))

    return resized_img


# Load the image (replace 'input_image.jpg' with your image file)
image = url_to_image("https://openclipart.org/image/2000px/202115")
# Example usage with the 'logo.png' file you downloaded earlier:
image = resize_image_with_aspect_ratio(
    image, desired_width=100, aspect_ratio=10/20)
# Check if the image was loaded properly
if image is None:
    print("Error: Image not found or unable to load.")
else:
    # Perform edge detection using Canny algorithm
    edges = cv2.Canny(image, threshold1=100, threshold2=200)

    # Convert edges to dot characters and display
    for y in range(edges.shape[0]):
        line = ''
        for x in range(edges.shape[1]):
            if edges[y, x]:
                line += '.'
            else:
                line += ' '
        print(line)
