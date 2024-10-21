import torchvision.transforms as transforms
import torchvision.models as models
import torch.nn as nn
import os
from django.shortcuts import render
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework.parsers import MultiPartParser, FormParser, JSONParser
from rest_framework import status
from django.core.files.uploadedfile import InMemoryUploadedFile
import cv2
import numpy as np
import urllib.request
import base64
import io

# Import Torch and Kornia
import torch
import kornia
from PIL import Image


# class ImageUploadView(APIView):
#     parser_classes = [MultiPartParser, FormParser, JSONParser]

#     def post(self, request, format=None):
#         # Extract data from the request
#         image_file = request.FILES.get('file', None)
#         image_url = request.data.get('url', None)
#         width = request.data.get('width', 100)
#         if width:
#             width = int(float(width))
#         else:
#             width = 100  # Set default width if not provided

#         if image_file:
#             image = self.read_image_file(image_file)
#         elif image_url:
#             image = self.read_image_url(image_url)
#         else:
#             return Response({'error': 'No image provided'}, status=status.HTTP_400_BAD_REQUEST)

#         # Generate ASCII art using Kornia edge detection
#         ascii_art, edge_image_b64, combined_image_b64 = self.generate_ascii_art(
#             image, width)

#         # Return the ASCII art and images
#         return Response({
#             'ascii_art': ascii_art,
#             'edge_image': edge_image_b64,
#             'combined_image': combined_image_b64
#         }, status=status.HTTP_200_OK)

#     def canny_generate_ascii_art(self, image, width):
#         # Resize image maintaining aspect ratio
#         aspect_ratio = image.shape[0] / image.shape[1]
#         height = int(width * aspect_ratio)
#         resized_img = cv2.resize(image, (width, height))

#         # Convert to RGB and Tensor
#         img_rgb = cv2.cvtColor(resized_img, cv2.COLOR_BGR2RGB)
#         img_tensor = self.image_to_tensor(
#             img_rgb).unsqueeze(0)  # Add batch dimension

#         # Edge detection using Kornia
#         edge_tensor = kornia.filters.Canny()(img_tensor.float() / 255.0)[0]
#         edges = edge_tensor[0]  # Get edge map

#         # Convert edge tensor to numpy array
#         edges_np = (edges.squeeze().numpy() * 255).astype(np.uint8)

#         # Give higher weight to the original image
#         gray_img = cv2.cvtColor(resized_img, cv2.COLOR_BGR2GRAY)
#         alpha = 0.7  # Weight for the original image (adjust as needed)
#         beta = 0.3   # Weight for the edges
#         combined = cv2.addWeighted(gray_img, alpha, edges_np, beta, 0)

#         # Normalize pixel values
#         normalized_img = combined / 255.0

#         # Map pixels to ASCII characters
#         ascii_art = self.map_pixels_to_ascii(normalized_img)

#         # Encode edge image and combined image to base64
#         edge_image_b64 = self.encode_image(edges_np)
#         combined_image_b64 = self.encode_image(combined)

#         return ascii_art, edge_image_b64, combined_image_b64

#     def canny_image_to_tensor(self, img):
#         # Convert a numpy array to a PyTorch tensor
#         img_tensor = torch.from_numpy(
#             img.transpose(2, 0, 1))  # Convert HWC to CHW
#         return img_tensor

#     def encode_image(self, img):
#         # Convert image to PIL Image and encode to base64
#         if len(img.shape) == 2:
#             img_pil = Image.fromarray(img)
#         else:
#             img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

#         buffered = io.BytesIO()
#         img_pil.save(buffered, format="PNG")
#         img_b64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
#         return img_b64

#     def map_pixels_to_ascii(self, normalized_img):
#         # Define a larger set of ASCII characters from dark to light
#         chars = np.array(
#             list('$@B%8&WM#*oahkbdpqwmZ0QLCJUYXzcvunxrjft/\|()1{}[]?-_+~<>i!lI;:,"^\'. '))
#         # Invert the normalized image if needed
#         normalized_img = 1 - normalized_img

#         # Map pixel values to indices of chars
#         indices = (normalized_img * (len(chars) - 1)).astype(int)

#         # Build the ASCII art line by line
#         ascii_art_lines = []
#         for row in indices:
#             line = ''.join(chars[row])
#             ascii_art_lines.append(line)

#         ascii_art = '\n'.join(ascii_art_lines)
#         return ascii_art

#     def read_image_file(self, image_file: InMemoryUploadedFile):
#         image_data = image_file.read()
#         image_array = np.frombuffer(image_data, np.uint8)
#         image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
#         return image

#     def read_image_url(self, url):
#         resp = urllib.request.urlopen(url)
#         image = np.asarray(bytearray(resp.read()), dtype="uint8")
#         image = cv2.imdecode(image, cv2.IMREAD_COLOR)
#         return image


# class ImageUploadView(APIView):

#     def hed_generate_ascii_art(self, image, width):
#         # Resize image maintaining aspect ratio
#         aspect_ratio = image.shape[0] / image.shape[1]
#         height = int(width * aspect_ratio)
#         resized_img = cv2.resize(image, (width, height))

#         # Convert to grayscale
#         gray_img = cv2.cvtColor(resized_img, cv2.COLOR_BGR2GRAY)

#         # Use HED edge detection
#         edges = self.hed_edge_detection(resized_img)

#         # Combine edges and grayscale image for more detail
#         combined = cv2.addWeighted(gray_img, 0.5, edges, 0.5, 0)

#         # Normalize pixel values
#         normalized_img = combined / 255.0

#         # Map pixels to ASCII characters
#         ascii_art = [self.map_pixels_to_ascii(
#             normalized_img), self.map_pixels_to_ascii(normalized_img, invert=True)]

#         return ascii_art

#     def hed_edge_detection(self, image):
#         # Load pre-trained HED model
#         proto_path = 'deploy.prototxt'
#         model_path = 'hed_pretrained_bsds.caffemodel'

#         # Download model files if they don't exist
#         if not os.path.isfile(proto_path) or not os.path.isfile(model_path):
#             self.download_hed_model(proto_path, model_path)

#         net = cv2.dnn.readNetFromCaffe(proto_path, model_path)

#         # Prepare the image for HED
#         blob = cv2.dnn.blobFromImage(image, scalefactor=1.0, size=image.shape[:2],
#                                      mean=(104.00698793, 116.66876762,
#                                            122.67891434),
#                                      swapRB=False, crop=False)

#         # Perform edge detection
#         net.setInput(blob)
#         hed = net.forward()
#         hed = hed[0, 0]  # Remove unnecessary dimensions

#         # Normalize to [0, 255]
#         hed = cv2.resize(hed, (image.shape[1], image.shape[0]))
#         hed = (255 * hed).astype(np.uint8)

#         return hed

#     def download_hed_model(self, proto_path, model_path):
#         # URLs for the model files
#         proto_url = 'https://github.com/s9xie/hed/blob/master/examples/hed/deploy.prototxt'
#         model_url = 'https://vcl.ucsd.edu/hed/hed_pretrained_bsds.caffemodel'

#         # Download and save the prototxt file
#         print("Downloading HED deploy.prototxt...")
#         response = urllib.request.urlopen(proto_url)
#         with open(proto_path, 'wb') as f:
#             f.write(response.read())

#         # Download and save the caffemodel file
#         print("Downloading HED hed_pretrained_bsds.caffemodel...")
#         response = urllib.request.urlopen(model_url)
#         with open(model_path, 'wb') as f:
#             f.write(response.read())


# class ImageUploadView(APIView):
#     parser_classes = [MultiPartParser, FormParser, JSONParser]

#     def gamma_correction_post(self, request, format=None):
#         # Extract data from the request
#         image_file = request.FILES.get('file', None)
#         image_url = request.data.get('url', None)
#         width = request.data.get('width', 50)
#         if width:
#             width = int(float(width))
#         else:
#             width = 50  # Set default width if not provided
#         if image_file:
#             image = self.read_image_file(image_file)
#         elif image_url:
#             image = self.read_image_url(image_url)
#         else:
#             return Response({'error': 'No image provided'}, status=status.HTTP_400_BAD_REQUEST)

#         aspect_ratio = image.shape[0] / image.shape[1]

#         # Resize image
#         height = int(width * aspect_ratio)
#         resized_img = cv2.resize(image, (width, height))

#         # Convert image to grayscale
#         gray_img = cv2.cvtColor(resized_img, cv2.COLOR_BGR2GRAY)

#         # Normalize pixel values to range 0-1
#         normalized_img = gray_img / 255.0

#         # Generate multiple variants with different parameters
#         gamma_values = [0.8, 1.0, 1.2]
#         chars = ['@', '%', '#', '*', '+', '=', '-',
#                  ':', '.', ' ']  # From darkest to lightest

#         variants = []
#         reconstructed_images = []
#         for gamma in gamma_values:
#             # Apply gamma correction
#             corrected_img = np.power(normalized_img, gamma)

#             # Map pixels to characters and reconstruct image
#             dot_art_lines = []
#             reconstructed_img = np.zeros_like(corrected_img)
#             for y in range(corrected_img.shape[0]):
#                 line = ''
#                 for x in range(corrected_img.shape[1]):
#                     pixel_value = corrected_img[y, x]
#                     index = int(pixel_value * (len(chars) - 1))
#                     index = len(chars) - index - 1  # Invert index
#                     index = np.clip(index, 0, len(chars) - 1)
#                     char = chars[index]
#                     line += char

#                     # Reconstruct the image from ASCII art
#                     reconstructed_img[y, x] = (
#                         len(chars) - index - 1) / (len(chars) - 1)

#                 dot_art_lines.append(line)
#             dot_art = '\n'.join(dot_art_lines)
#             variants.append({
#                 'gamma': gamma,
#                 'dot_art': dot_art,
#                 'reconstructed_img': reconstructed_img
#             })

#         # Evaluate variants using perceptual loss
#         best_variant = self.select_best_variant(resized_img, variants)

#         # Return the best variant
#         return Response({
#             'dot_art': best_variant['dot_art']
#         }, status=status.HTTP_200_OK)

#     def select_best_variant(self, original_img, variants):
#         # Convert images to tensors
#         transform = transforms.Compose([
#             transforms.ToTensor(),
#             transforms.Resize((224, 224)),  # Resize to match VGG input size
#         ])

#         # Ensure original_img is in the correct format and type
#         original_img_rgb = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
#         original_img_rgb = original_img_rgb.astype(
#             np.float32) / 255.0  # Normalize to [0,1]
#         original_tensor = transform(original_img_rgb).unsqueeze(
#             0)  # Add batch dimension

#         # Load pre-trained VGG19 model
#         vgg = models.vgg19(pretrained=True).features.eval()

#         # Use GPU if available
#         device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#         vgg.to(device)
#         original_tensor = original_tensor.to(device)

#         # Define loss function
#         criterion = nn.MSELoss()

#         min_loss = float('inf')
#         best_variant = None

#         for variant in variants:
#             reconstructed_img = variant['reconstructed_img']
#             # Convert reconstructed image to tensor
#             reconstructed_img_rgb = np.stack((reconstructed_img,)*3, axis=-1)
#             reconstructed_img_rgb = reconstructed_img_rgb.astype(
#                 np.float32)  # Ensure type is float32
#             reconstructed_tensor = transform(
#                 reconstructed_img_rgb).unsqueeze(0).to(device)

#             # Get feature maps
#             with torch.no_grad():
#                 original_features = vgg(original_tensor)
#                 reconstructed_features = vgg(reconstructed_tensor)

#             # Calculate perceptual loss
#             loss = criterion(original_features, reconstructed_features).item()

#             if loss < min_loss:
#                 min_loss = loss
#                 best_variant = variant

#         return best_variant


# def main_post(self, request, format=None):
#     # Extract data from the request
#     image_file = request.FILES.get('file', None)
#     image_url = request.data.get('url', None)
#     width = request.data.get('width', 50)
#     if width:
#         width = int(float(width))
#     aspect_ratio = request.data.get('aspect_ratio', None)
#     if aspect_ratio:
#         aspect_ratio = float(aspect_ratio)
#     else:
#         aspect_ratio = None  # Ensure it's None if not provided
#     if image_file:
#         image = self.read_image_file(image_file)
#     elif image_url:
#         image = self.read_image_url(image_url)
#     else:
#         return Response({'error': 'No image provided'}, status=status.HTTP_400_BAD_REQUEST)

#     if aspect_ratio is None:
#         aspect_ratio = image.shape[0] / image.shape[1]

#     # Resize image
#     height = int(width * aspect_ratio)
#     resized_img = cv2.resize(image, (width, height))

#     # Convert image to grayscale
#     gray_img = cv2.cvtColor(resized_img, cv2.COLOR_BGR2GRAY)

#     # Normalize pixel values to range 0-1
#     normalized_img = gray_img / 255.0

#     # Generate multiple variants with different gamma values
#     # You can adjust these values
#     gamma_values = [0.5, 0.8, 1.0, 1.2, 1.5, 2.0]
#     chars = ['@', '%', '#', '*', '+', '=', '-',
#              ':', '.', ' ']  # From darkest to lightest

#     variants = []
#     for gamma in gamma_values:
#         # Apply gamma correction
#         corrected_img = np.power(normalized_img, gamma)

#         # Map pixels to characters
#         dot_art_lines = []
#         for y in range(corrected_img.shape[0]):
#             line = ''
#             for x in range(corrected_img.shape[1]):
#                 pixel_value = corrected_img[y, x]
#                 index = int(pixel_value * (len(chars) - 1))
#                 # Invert index to match characters from dark to light
#                 index = len(chars) - index - 1
#                 line += chars[index]
#             dot_art_lines.append(line)
#         dot_art = '\n'.join(dot_art_lines)
#         variants.append({
#             'gamma': gamma,
#             'dot_art': dot_art
#         })

#     # Return the variants

# Import Torch and Kornia


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

        # Generate ASCII art using enhanced edge processing
        ascii_art_variants, edge_image_b64, combined_image_b64, edge_blurred_base64, high_contrast_base64, gray_img_b64 = self.generate_ascii_art(
            image, width)

        # Return the ASCII art and images
        return Response({
            'ascii_art_variants': ascii_art_variants,
            'edge_image': edge_image_b64,
            'combined_image': combined_image_b64,
            "edge_blurred_base64": edge_blurred_base64,
            "high_contrast_base64": high_contrast_base64,
            "gray_img_b64": gray_img_b64
        }, status=status.HTTP_200_OK)

    def generate_ascii_art(self, image, width):
        # Resize image maintaining aspect ratio
        aspect_ratio = image.shape[0] / image.shape[1]
        height = int(width * aspect_ratio)
        resized_img = cv2.resize(image, (width, height))

        # Convert to grayscale
        gray_img = cv2.cvtColor(resized_img, cv2.COLOR_BGR2GRAY)

        # Perform edge detection using Kornia
        img_rgb = cv2.cvtColor(resized_img, cv2.COLOR_BGR2RGB)
        img_tensor = self.image_to_tensor(img_rgb).unsqueeze(
            0).float() / 255.0  # Convert to float tensor in [0,1]

        # Use Kornia for edge detection
        edge_detector = kornia.filters.Canny()
        edges_tuple = edge_detector(img_tensor)
        edges = edges_tuple[0]  # Edge map
        edges_np = edges.squeeze().cpu().numpy()
        edges_np = 1 - edges_np  # Convert to numpy array
        edges_np_norm = (edges_np - (np.min(edges_np))) / \
            ((np.max(edges_np) - np.min(edges_np)))

        # Compute edge density map by blurring the edge map
        edges_blurred = cv2.GaussianBlur(edges_np, (5, 5), 0)
        edges_blurred_norm = edges_blurred / edges_blurred.max()

        # Use edge density to adjust the contrast of the grayscale image
        # Create a high-contrast version of the grayscale image using CLAHE
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        high_contrast = clahe.apply(gray_img)
        high_contrast_norm = high_contrast / 255.0

        # Normalize the original grayscale image
        gray_norm = gray_img / 255.0
        import copy

        # Enhance edges by darkening pixels where edges are detected
        # Binarize the edge map
        _, edge_binary = cv2.threshold(
            (edges_np * 255).astype(np.uint8), 100, 1, cv2.THRESH_BINARY)
        # Set pixels to black where edges are detected
        # adjusted_img[edge_binary == 0] = 0
        # darken_edged_image = copy.deepcopy(gray_norm)
        # darken_edged_image[edge_binary == 0] = 0

        darkened_edge_high_contrast = copy.deepcopy(high_contrast_norm)
        # combined_image = edges_blurred_norm * 0.3 + high_contrast_norm * 0.7
        """
        Darkens the high-contrast image based on the edge density map.
        
        The `edges_blurred_norm` variable represents a normalized edge density map, where values closer to 1 indicate stronger edges. This function selects the appropriate value to use for each pixel in the `darkened_edge_high_contrast` image:
        
        - If the edge density is less than 0.5, the edge density value is used.
        - Otherwise, the original `darkened_edge_high_contrast` value is used.
        
        This effectively darkens the pixels in the high-contrast image where edges are detected, enhancing the visual contrast of the edges.
        """
        # darkened_edge_high_contrast = np.where(
        #     edges_blurred_norm < 0.3, edges_blurred_norm, darkened_edge_high_contrast)
        for row in range(edges_np.shape[0]):
            for col in range(edges_np.shape[1]):

                darkened_edge_high_contrast[row,

                                            col] = edges_np_norm[row, col] if edges_np_norm[row, col] < 0.8 else darkened_edge_high_contrast[row, col]
        # darkened_edge_high_contrast = np.add(
        #     darkened_edge_high_contrast * 0.7, edges_np * 0.3)

        # weighted_sum_high_contrast_and_edge = copy.deepcopy(high_contrast_norm)
        # for y in range(high_contrast_norm.shape[0]):
        #     for x in range(high_contrast_norm.shape[1]):
        #         weighted_sum_high_contrast_and_edge[y, x] = weighted_sum_high_contrast_and_edge[y, x]
        #         index = int(pixel_value * (len(chars) - 1))
        #         index = len(chars) - index - 1
        #         line += chars[index]
        # weighted_sum_high_contrast_and_edge

        # Map pixels to ASCII characters
        # ascii_art_variants = self.generate_variants(adjusted_img, gray_img)
        ascii_art_variants = self.generate_variants(
            darkened_edge_high_contrast)
        edge_blurred_base64 = self.encode_image(
            (edges_blurred_norm * 255).astype(np.uint8))
        high_contrast_base64 = self.encode_image(
            (high_contrast_norm * 255).astype(np.uint8))
        # Encode edge image and adjusted image to base64
        edge_image_b64 = self.encode_image(
            (edges_np_norm * 255).astype(np.uint8))
        combined_image_b64 = self.encode_image(
            (darkened_edge_high_contrast * 255).astype(np.uint8))
        gray_img_b64 = self.encode_image(
            (gray_img).astype(np.uint8))

        ascii_art_variants.append({
            'gamma': "Gray Image",
            'dot_art': self.map_pixels_to_ascii(gray_norm)
        })
        ascii_art_variants.append({
            'gamma': "High Contrast Image",
            'dot_art': self.map_pixels_to_ascii(high_contrast_norm)
        })

        return ascii_art_variants, edge_image_b64, combined_image_b64, edge_blurred_base64, high_contrast_base64, gray_img_b64

    def image_to_tensor(self, img):
        # Convert a numpy array to a PyTorch tensor
        img_tensor = torch.from_numpy(
            img.transpose(2, 0, 1))  # Convert HWC to CHW
        return img_tensor

    def encode_image(self, img):
        # Convert image to PIL Image and encode to base64
        if len(img.shape) == 2:
            img_pil = Image.fromarray(img)
        else:
            img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

        buffered = io.BytesIO()
        img_pil.save(buffered, format="PNG")
        img_b64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
        return img_b64

    def map_pixels_to_ascii(self, normalized_img):
        # Define a set of ASCII characters from dark to light
        chars = ['@', '%', '#', '*', '+', '=', '-',
                 ':', '.', ' ']

        # Map pixels to characters
        dot_art_lines = []
        for y in range(normalized_img.shape[0]):
            line = ''
            for x in range(normalized_img.shape[1]):
                pixel_value = normalized_img[y, x]
                index = int(pixel_value * (len(chars) - 1))
                # Invert index to match characters from dark to light
                index = len(chars) - index - 1
                line += chars[index]
            dot_art_lines.append(line)
        dot_art = '\n'.join(dot_art_lines)
        return dot_art

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

    def generate_variants(self, normalized_img):
        # Normalize pixel values to range 0-1
        # Generate multiple variants with different gamma values
        # You can adjust these values
        gamma_values = [0.5, 0.8, 1.0, 1.2, 1.5, 2.0]
        chars = ['@', '%', '#', '*', '+', '=', '-',
                 ':', '.', ' ']

        variants = []
        for gamma in gamma_values:
            # Apply gamma correction
            corrected_img = np.sign(normalized_img) * \
                np.power(np.abs(normalized_img), gamma)

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
        return variants

    # def map_pixels_to_ascii(self, normalized_img):
    #     # Define a larger set of ASCII characters from dark to light
    #     chars = np.array(
    #         list('$@B%8&WM#*oahkbdpqwmZ0QLCJUYXzcvunxrjft/\|()1{}[]?-_+~<>i!lI;:,"^\'. '))
    #     # Invert the normalized image if needed
    #     normalized_img = 1 - normalized_img

    #     # Map pixel values to indices of chars
    #     indices = (normalized_img * (len(chars) - 1)).astype(int)

    #     # Build the ASCII art line by line
    #     ascii_art_lines = []
    #     for row in indices:
    #         line = ''.join(chars[row])
    #         ascii_art_lines.append(line)

    #     ascii_art = '\n'.join(ascii_art_lines)
    #     return ascii_art
