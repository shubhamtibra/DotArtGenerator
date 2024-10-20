from django.shortcuts import render
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework.parsers import MultiPartParser, FormParser, JSONParser
from rest_framework import status
from django.core.files.uploadedfile import InMemoryUploadedFile
from django.core.files.storage import default_storage
import cv2
import numpy as np
import urllib.request


class ImageUploadView(APIView):
    parser_classes = [MultiPartParser, FormParser, JSONParser]

    def post(self, request, format=None):
        image_file = request.FILES.get('file', None)
        image_url = request.data.get('url', None)
        width = int(request.data.get('width', 50))
        aspect_ratio = request.data.get('aspect_ratio', None)
        if aspect_ratio:
            aspect_ratio = float(aspect_ratio)
        else:
            aspect_ratio = None

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

        # Edge detection
        edges = cv2.Canny(resized_img, threshold1=100, threshold2=200)

        # Convert edges to dot characters
        dot_art_lines = []
        for y in range(edges.shape[0]):
            line = ''
            for x in range(edges.shape[1]):
                if edges[y, x]:
                    line += '.'
                else:
                    line += ' '
            dot_art_lines.append(line)

        # Return dot art as a string
        dot_art = '\n'.join(dot_art_lines)
        return Response({'dot_art': dot_art}, status=status.HTTP_200_OK)

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
