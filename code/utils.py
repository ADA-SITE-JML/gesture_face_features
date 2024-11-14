# utils.py
import os
from PIL import Image, ImageEnhance
import random
import matplotlib.pyplot as plt
import numpy as np
from google.colab.patches import cv2_imshow
import random
import requests
from io import BytesIO


def plot_img_grid(images, imgs_per_row=5, labels=None, figsize=(12, 8)):
    """
    Plots a grid of images with a specified number of images per row.
    
    Parameters:
    - images: List of image arrays.
    - imgs_per_row: Number of images per row.
    - labels: List of labels for each image (optional).
    - figsize: Tuple to set the figure size.
    """
    num_images = len(images)
    num_rows = (num_images + imgs_per_row - 1) // imgs_per_row  # Calculate the number of rows needed
    
    fig, axes = plt.subplots(num_rows, imgs_per_row, figsize=figsize)
    fig.subplots_adjust(hspace=0.1, wspace=0.1)  # Adjust spacing between images
    
    # Flatten axes array to ensure it’s a list, even if there’s only one row
    axes = axes.flatten() if isinstance(axes, np.ndarray) else [axes]
    
    for i, ax in enumerate(axes):
        if i < num_images:
            ax.imshow(images[i])
            ax.axis('off')  # Hide axis
            if labels is not None:
                ax.set_title(labels[i], fontsize=10)
        else:
            ax.axis('off')  # Turn off any unused subplots
    
    plt.show()


def plot_imgs(imgs, imgs_per_row=1):
    if isinstance(imgs, str):  # Check if it's a folder path
        img_paths = [os.path.join(imgs, img_name) for img_name in os.listdir(imgs)]
        num_imgs = len(img_paths)
        img_loader = lambda idx: Image.open(img_paths[idx])
    elif isinstance(imgs, list) and all(isinstance(img, Image.Image) for img in imgs):  # Check if it's a list of PIL images
        num_imgs = len(imgs)
        img_loader = lambda idx: imgs[idx]
    else:
        raise ValueError("Invalid input. Provide either a folder path or a list of PIL images.")

    num_rows = (num_imgs + imgs_per_row - 1) // imgs_per_row

    fig, axs = plt.subplots(num_rows, imgs_per_row, figsize=(4 * imgs_per_row, 4 * num_rows))
    axs = axs.flatten()

    for idx in range(num_imgs):
        try:
            img = img_loader(idx)
            show_img(img, axs[idx], title=f'Image {idx + 1}')
        except Exception as e:
            print(f"Error loading image: {e}")

    plt.show()


def show_img(img, title=None, ax=None):
    if ax is None:
        fig, ax = plt.subplots()
    ax.imshow(np.array(img))
    if title:
        ax.set_title(title)
    ax.axis('off')
    if ax is None:
        plt.show()


def load_imgs(img_paths, shuffle=False):
    if shuffle:
        random.shuffle(img_paths)

    loaded_imgs = []
    for path in img_paths:
        # Extract the last part of the path (letter)
        folder_name = os.path.basename(os.path.dirname(path)).lower()

        try:
            img = Image.open(path)
            loaded_imgs.append(img)
        except Exception as e:
            print(f"Error loading img from {path}: {e}")

    print(f"{len(loaded_imgs)} images loaded")

    return loaded_imgs


def process_img(input_img):
    if isinstance(input_img, str):  # Check if the input is an image path
        img = Image.open(input_img)
    elif isinstance(input_img, Image.Image):  # Check if the input is a PIL Image
        img = input_img
    else:
        raise ValueError("Unsupported input type. Please provide either an image path or a PIL Image.")
    return img

import os
import re

def get_paths(root_directory, letter=None, img_count=None, img_numbers_list=None):
    folder_paths = []
    img_paths = []
    labels = []
    img_numbers = []  # To store extracted image numbers

    # Get the list of folders in the root directory
    folders = os.listdir(root_directory)

    # Filter by starting letter(s) if provided
    if letter is not None:
        if isinstance(letter, str):
            letter = [letter.upper()]
        else:
            letter = [l.upper() for l in letter]
        folders = [folder for folder in folders if folder.upper().startswith(tuple(letter))]

    if not folders:
        print("No folders found matching the criteria.")
        return folder_paths, img_paths, labels, img_numbers  # Return empty lists if no folders match

    count = 0  # Counter to keep track of how many images have been selected

    for folder in folders:
        folder_path = os.path.join(root_directory, folder)
        folder_paths.append(folder_path)

        files = os.listdir(folder_path)

        for img in files:
            if img_count is not None and count >= img_count:
                break  # Stop if we've already selected img_count images

            img_path = os.path.join(folder_path, img)

            # Extract the image number using regex
            match = re.search(r'IMG_(\d+)\.JPG', img)
            if match:
                img_number = int(match.group(1))  # Get the image number as an integer
                if img_numbers_list is None or img_number in img_numbers_list:
                    img_paths.append(img_path)
                    labels.append(folder)  # Add folder name as label for each found image
                    img_numbers.append(img_number)  # Store the extracted image number
                    count += 1  # Increment the counter
            else:
                img_numbers.append(None)  # Append None if no number is found

    return folder_paths, img_paths, labels, img_numbers




def fetch_img(url):
    try:
        response = requests.get(url)
        if response.status_code == 200:
            img = Image.open(BytesIO(response.content))
            return img
        else:
            print(f"Failed to retrieve image. Status code: {response.status_code}")
            return None
    except Exception as e:
        print(f"An error occurred: {e}")
        return None



