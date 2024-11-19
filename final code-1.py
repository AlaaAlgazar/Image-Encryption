import numpy as np
from scipy.fft import dct, idct
import matplotlib.pyplot as plt
from PIL import Image
import tkinter as tk
from tkinter import filedialog
import time

# Logistic map function for generating chaotic sequence
def logistic_map(r, x, n):
    chaotic_sequence = []
    for _ in range(n):
        x = r * x * (1 - x)
    for _ in range(n):
        x = r * x * (1 - x)
        chaotic_sequence.append(x)
    return chaotic_sequence

# Simple Discrete Cosine Transform (DCT) encryption with edge padding
def simple_dct_encryption(image, block, s,r,x0):
    # Pad image to ensure its dimensions are divisible by block_size and edges are captured
    block_length, block_width = block
    padded_height = ((image.shape[0] + block_length - 1) // block_length) * block_length
    padded_width = ((image.shape[1] + block_width - 1) // block_width) * block_width
    padded_image = np.pad(image, ((0, padded_height - image.shape[0]), 
                                   (0, padded_width - image.shape[1]), 
                                   (0,0)), mode='edge')  # Apply edge padding
    
    # Generate chaotic key matching padded image dimensions
    key_size = (padded_height // block_length, padded_width // block_width)
    chaotic_sequence = logistic_map(r, x0, key_size[0] * key_size[1])
    key = np.array(chaotic_sequence).reshape(key_size)
    key = key * s 
    # Initialize encrypted image
    encrypted_image = np.zeros_like(padded_image, dtype=float)

    # Apply DCT to image blocks for each channel
    for c in range(padded_image.shape[2]):
        for i in range(0, padded_image.shape[0], block_length):
            for j in range(0, padded_image.shape[1], block_width):
                block = padded_image[i:i+block_length, j:j+block_width, c]
                # Apply DCT to the block
                encrypted_block = dct(block, norm='ortho')
                # Apply chaotic sequence to the DCT coefficients
                encrypted_block *= key[i//block_length, j//block_width]
                encrypted_image[i:i+block_length, j:j+block_width, c] = encrypted_block

    return encrypted_image, key, key_size  # Return encrypted image and key

# Decryption using inverse DCT
def dct_decryption(encrypted_image, block, s,original_shape,r,x0):
    # Initialize decrypted image
    decrypted_image = np.zeros_like(encrypted_image, dtype=float)
    
    # Define the block size
    block_length, block_width = block
    # define key size
    padded_height = ((original_shape[0] + block_length - 1) // block_length) * block_length
    padded_width = ((original_shape[1] + block_width - 1) // block_width) * block_width
    key_size = (padded_height // block_length, padded_width // block_width)


    chaotic_sequence = logistic_map(r, x0, key_size[0] * key_size[1])
    key = np.array(chaotic_sequence).reshape(key_size)
    key = key * s 
    # Apply inverse DCT to encrypted blocks for each channel
    for c in range(encrypted_image.shape[2]):
        for i in range(0, encrypted_image.shape[0], block_length):
            for j in range(0, encrypted_image.shape[1], block_width):
                block = encrypted_image[i:i+block_length, j:j+block_width, c]
                # Divide by chaotic sequence to recover original DCT coefficients
                decrypted_block = block / key[i//block_length, j//block_width]
                # Apply inverse DCT to the block
                decrypted_block = idct(decrypted_block, norm='ortho')
                decrypted_image[i:i+block_length, j:j+block_width, c] = decrypted_block
    decrypted_image = decrypted_image[:original_shape[0], :original_shape[1], :]
    return decrypted_image

def compute_metrics(image):
    # Histogram
    histogram = np.histogram(image.flatten(), bins=256, range=(0,256))[0]

    # Entropy
    probs = histogram / float(np.sum(histogram))
    entropy = -np.sum(probs * np.log2(probs + 1e-10))

    # Correlation
    correlation = np.corrcoef(image[:,:,0].flatten(), image[:,:,1].flatten())[0,1]

    return histogram, entropy, correlation

# Create a Tkinter root window
root = tk.Tk()
root.withdraw()  # Hide the root window

# Ask user to choose image file
file_path = filedialog.askopenfilename(title="Choose an image file")

if file_path:
    # Load image
    image = np.array(Image.open(file_path))

    block = (8, 8)  # Define block size as (length, width)
    scalar_const = 256
    r = 3.6
    x0= 0.3
    # Encryption
    start_time_encryption = time.time()
    encrypted_image, key, key_size = simple_dct_encryption(image, block, scalar_const,r,x0)
    end_time_encryption = time.time()
    encryption_time = end_time_encryption - start_time_encryption

    # Decryption
    start_time_decryption = time.time()
    decrypted_image = dct_decryption(encrypted_image, block, scalar_const,image.shape,r,x0)
    end_time_decryption = time.time()
    decryption_time = end_time_decryption - start_time_decryption

    # Clip values to 0-255 range
    decrypted_image = np.clip(decrypted_image, 0, 255)

    # Convert to uint8
    decrypted_image = decrypted_image.astype(np.uint8)

    # Compute metrics for encrypted
    encrypted_histogram, encrypted_entropy, encrypted_correlation = compute_metrics(encrypted_image.astype(np.uint8))
    # Save encrypted image in the same format as the original image
    encrypted_image_pil = Image.fromarray(encrypted_image.astype(np.uint8))
    encrypted_image_pil.save('encrypted_image.' + Image.open(file_path).format.lower())

    # Save decrypted image in the same format as the original image
    decrypted_image_pil = Image.fromarray(decrypted_image)
    decrypted_image_pil.save('decrypted_image.' + Image.open(file_path).format.lower())
    
    print("Encrypted and decrypted images saved successfully.")
    print("Encryption time",encryption_time)
    print("Decryption time",decryption_time) 
    print("Entropy",encrypted_entropy) 
    print("Color correlation",encrypted_correlation) 
else:
    print("No image selected.")