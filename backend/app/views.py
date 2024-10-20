from django.shortcuts import render
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework.parsers import MultiPartParser, FormParser, JSONParser
from rest_framework import status
from django.core.files.uploadedfile import InMemoryUploadedFile
import cv2
import numpy as np
import urllib.request


class ImageUploadView(APIView):
    parser_classes = [MultiPartParser, FormParser, JSONParser]

    def post(self, request, format=None):
        # Extract data from the request
        image_file = request.FILES.get('file', None)
        image_url = request.data.get('url', None)
        width = request.data.get('width', 50)
        if width:
            width = int(float(width))
        aspect_ratio = request.data.get('aspect_ratio', None)
        if aspect_ratio:
            aspect_ratio = float(aspect_ratio)
        else:
            aspect_ratio = None  # Ensure it's None if not provided
        if image_file:
            image = self.read_image_file(image_file)
        elif image_url:
            image = self.read_image_url(image_url)
        else:
            return Response({'error': 'No image provided'}, status=status.HTTP_400_BAD_REQUEST)

        if aspect_ratio is None:
            aspect_ratio = image.shape[0] / image.shape[1]

        # Resize image
        height = int(width * aspect_ratio)
        resized_img = cv2.resize(image, (width, height))

        # Convert image to grayscale
        gray_img = cv2.cvtColor(resized_img, cv2.COLOR_BGR2GRAY)

        # Normalize pixel values to range 0-1
        normalized_img = gray_img / 255.0

        # Generate multiple variants with different gamma values
        # You can adjust these values
        gamma_values = [0.5, 0.8, 1.0, 1.2, 1.5, 2.0]
        chars = ['@', '%', '#', '*', '+', '=', '-',
                 ':', '.', ' ']  # From darkest to lightest

        variants = []
        for gamma in gamma_values:
            # Apply gamma correction
            corrected_img = np.power(normalized_img, gamma)

            # Map pixels to characters
            dot_art_lines = []
            for y in range(corrected_img.shape[0]):
                line = ''
                for x in range(corrected_img.shape[1]):
                    pixel_value = corrected_img[y, x]
                    index = int(pixel_value * (len(chars) - 1))
                    # Invert index to match characters from dark to light
                    index = len(chars) - index - 1
                    line += chars[index]
                dot_art_lines.append(line)
            dot_art = '\n'.join(dot_art_lines)
            variants.append({
                'gamma': gamma,
                'dot_art': dot_art
            })

        # Return the variants
        return Response({'variants': variants}, status=status.HTTP_200_OK)

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
