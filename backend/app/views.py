from django.shortcuts import render
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework.parsers import MultiPartParser, FormParser, JSONParser
from rest_framework import status
from django.core.files.uploadedfile import InMemoryUploadedFile
import cv2
import numpy as np
import urllib.request

# Import PyTorch and related modules
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms


class ImageUploadView(APIView):
    parser_classes = [MultiPartParser, FormParser, JSONParser]

    def post(self, request, format=None):
        # Extract data from the request
        image_file = request.FILES.get('file', None)
        image_url = request.data.get('url', None)
        width = request.data.get('width', 50)
        if width:
            width = int(float(width))
        else:
            width = 50  # Set default width if not provided
        if image_file:
            image = self.read_image_file(image_file)
        elif image_url:
            image = self.read_image_url(image_url)
        else:
            return Response({'error': 'No image provided'}, status=status.HTTP_400_BAD_REQUEST)

        aspect_ratio = image.shape[0] / image.shape[1]

        # Resize image
        height = int(width * aspect_ratio)
        resized_img = cv2.resize(image, (width, height))

        # Convert image to grayscale
        gray_img = cv2.cvtColor(resized_img, cv2.COLOR_BGR2GRAY)

        # Normalize pixel values to range 0-1
        normalized_img = gray_img / 255.0

        # Generate multiple variants with different parameters
        gamma_values = [0.8, 1.0, 1.2]
        chars = ['@', '%', '#', '*', '+', '=', '-',
                 ':', '.', ' ']  # From darkest to lightest

        variants = []
        reconstructed_images = []
        for gamma in gamma_values:
            # Apply gamma correction
            corrected_img = np.power(normalized_img, gamma)

            # Map pixels to characters and reconstruct image
            dot_art_lines = []
            reconstructed_img = np.zeros_like(corrected_img)
            for y in range(corrected_img.shape[0]):
                line = ''
                for x in range(corrected_img.shape[1]):
                    pixel_value = corrected_img[y, x]
                    index = int(pixel_value * (len(chars) - 1))
                    index = len(chars) - index - 1  # Invert index
                    index = np.clip(index, 0, len(chars) - 1)
                    char = chars[index]
                    line += char

                    # Reconstruct the image from ASCII art
                    reconstructed_img[y, x] = (
                        len(chars) - index - 1) / (len(chars) - 1)

                dot_art_lines.append(line)
            dot_art = '\n'.join(dot_art_lines)
            variants.append({
                'gamma': gamma,
                'dot_art': dot_art,
                'reconstructed_img': reconstructed_img
            })

        # Evaluate variants using perceptual loss
        best_variant = self.select_best_variant(resized_img, variants)

        # Return the best variant
        return Response({
            'dot_art': best_variant['dot_art']
        }, status=status.HTTP_200_OK)

    def select_best_variant(self, original_img, variants):
        # Convert images to tensors
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((224, 224)),  # Resize to match VGG input size
        ])

        # Ensure original_img is in the correct format and type
        original_img_rgb = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
        original_img_rgb = original_img_rgb.astype(
            np.float32) / 255.0  # Normalize to [0,1]
        original_tensor = transform(original_img_rgb).unsqueeze(
            0)  # Add batch dimension

        # Load pre-trained VGG19 model
        vgg = models.vgg19(pretrained=True).features.eval()

        # Use GPU if available
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        vgg.to(device)
        original_tensor = original_tensor.to(device)

        # Define loss function
        criterion = nn.MSELoss()

        min_loss = float('inf')
        best_variant = None

        for variant in variants:
            reconstructed_img = variant['reconstructed_img']
            # Convert reconstructed image to tensor
            reconstructed_img_rgb = np.stack((reconstructed_img,)*3, axis=-1)
            reconstructed_img_rgb = reconstructed_img_rgb.astype(
                np.float32)  # Ensure type is float32
            reconstructed_tensor = transform(
                reconstructed_img_rgb).unsqueeze(0).to(device)

            # Get feature maps
            with torch.no_grad():
                original_features = vgg(original_tensor)
                reconstructed_features = vgg(reconstructed_tensor)

            # Calculate perceptual loss
            loss = criterion(original_features, reconstructed_features).item()

            if loss < min_loss:
                min_loss = loss
                best_variant = variant

        return best_variant

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
