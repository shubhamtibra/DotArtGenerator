from django.shortcuts import render
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework.parsers import MultiPartParser, FormParser, JSONParser
from rest_framework import status
from django.core.files.uploadedfile import InMemoryUploadedFile
import cv2
import numpy as np
import urllib.request
import os


class ImageUploadView(APIView):
    parser_classes = [MultiPartParser, FormParser, JSONParser]

    def post(self, request, format=None):
        # Extract data from the request
        image_file = request.FILES.get('file', None)
        image_url = request.data.get('url', None)
        width = request.data.get('width', 100)
        if width:
            width = int(float(width))
        else:
            width = 100  # Set default width if not provided

        if image_file:
            image = self.read_image_file(image_file)
        elif image_url:
            image = self.read_image_url(image_url)
        else:
            return Response({'error': 'No image provided'}, status=status.HTTP_400_BAD_REQUEST)

        # Generate ASCII art using HED edge detection
        ascii_art = self.generate_ascii_art(image, width)

        # Return the ASCII art
        return Response({
            'ascii_art': ascii_art
        }, status=status.HTTP_200_OK)

    def generate_ascii_art(self, image, width):
        # Resize image maintaining aspect ratio
        aspect_ratio = image.shape[0] / image.shape[1]
        height = int(width * aspect_ratio)
        resized_img = cv2.resize(image, (width, height))

        # Convert to grayscale
        gray_img = cv2.cvtColor(resized_img, cv2.COLOR_BGR2GRAY)

        # Use HED edge detection
        edges = self.hed_edge_detection(resized_img)

        # Combine edges and grayscale image for more detail
        combined = cv2.addWeighted(gray_img, 0.5, edges, 0.5, 0)

        # Normalize pixel values
        normalized_img = combined / 255.0

        # Map pixels to ASCII characters
        ascii_art = self.map_pixels_to_ascii(normalized_img)

        return ascii_art

    def hed_edge_detection(self, image):
        # Load pre-trained HED model
        proto_path = 'deploy.prototxt'
        model_path = 'hed_pretrained_bsds.caffemodel'

        # Download model files if they don't exist
        if not os.path.isfile(proto_path) or not os.path.isfile(model_path):
            self.download_hed_model(proto_path, model_path)

        net = cv2.dnn.readNetFromCaffe(proto_path, model_path)

        # Prepare the image for HED
        blob = cv2.dnn.blobFromImage(image, scalefactor=1.0, size=image.shape[:2],
                                     mean=(104.00698793, 116.66876762,
                                           122.67891434),
                                     swapRB=False, crop=False)

        # Perform edge detection
        net.setInput(blob)
        hed = net.forward()
        hed = hed[0, 0]  # Remove unnecessary dimensions

        # Normalize to [0, 255]
        hed = cv2.resize(hed, (image.shape[1], image.shape[0]))
        hed = (255 * hed).astype(np.uint8)

        return hed

    def map_pixels_to_ascii(self, normalized_img):
        # Define ASCII characters from dark to light
        chars = np.array(list("@%#*+=-:. "))

        # Map pixel values to indices of chars
        indices = (normalized_img * (len(chars) - 1)).astype(int)

        # Build the ASCII art line by line
        ascii_art_lines = []
        for row in indices:
            line = ''.join(chars[row])
            ascii_art_lines.append(line)

        ascii_art = '\n'.join(ascii_art_lines)
        return ascii_art

    # def download_hed_model(self, proto_path, model_path):
    #     # URLs for the model files
    #     #proto_url = 'https://raw.githubusercontent.com/opencv/opencv_extra/master/testdata/dnn/hed/deploy.prototxt'
    #     model_url = 'http://vcl.ucsd.edu/hed/hed_pretrained_bsds.caffemodel'

    #     # Download and save the prototxt file
    #     #print("Downloading HED deploy.prototxt...")
    #     #response = urllib.request.urlopen(proto_url)
    #     #with open(proto_path, 'wb') as f:
    #     #    f.write(response.read())

    #     # Download and save the caffemodel file
    #     print("Downloading HED hed_pretrained_bsds.caffemodel...")
    #     response = urllib.request.urlopen(model_url)
    #     with open(model_path, 'wb') as f:
    #         f.write(response.read())

    def read_image_file(self, image_file: InMemoryUploadedFile):
        image_data = image_file.read()
        image_array = np.frombuffer(image_data, np.uint8)
        image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
        return image

    def read_image_url(self, url):
        resp = urllib.request.urlopen(url)
        image = np.asarray(bytearray(resp.read()), dtype="uint8")
        image = cv2.imdecode(image, cv2.IMREAD_COLOR)
        return image
