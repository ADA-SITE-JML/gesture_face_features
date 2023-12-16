# utils.py
import os
from PIL import Image
import random
import matplotlib.pyplot as plt
import numpy as np
from google.colab.patches import cv2_imshow


def plot_imgs(folder_path):
    img_paths = [os.path.join(folder_path, img_name) for img_name in os.listdir(folder_path)]

    for i, img_path in enumerate(img_paths):
        try:
            img = Image.open(img_path)
            show_img(img, title=f'Image {i+1}')
        except Exception as e:
            print(f"Error loading img from {img_path}: {e}")


def show_img(img, title=None):
    img_array = np.array(img)
    plt.imshow(img_array)
    if title:
        plt.title(title)
    plt.axis('off')
    plt.show()


def load_imgs(img_paths, shuffle=False, letter=None):
    if shuffle:
        random.shuffle(img_paths)

    loaded_imgs = []
    for path in img_paths:
        if letter is not None and not path.lower().endswith(f"/{letter.lower()}"):
            continue

        try:
            img = Image.open(path)
            loaded_imgs.append(img)
        except Exception as e:
            print(f"Error loading img from {path}: {e}")

    return loaded_imgs


def process_img(input_img):
    if isinstance(input_img, str):  # Check if the input is an image path
        img = Image.open(input_img)
    elif isinstance(input_img, Image.Image):  # Check if the input is a PIL Image
        img = input_img
    else:
        raise ValueError("Unsupported input type. Please provide either an image path or a PIL Image.")
    return img


def get_paths(root_directory, letter=None):
    folder_paths = []
    img_paths = []

    folders = os.listdir(root_directory)
    for folder in folders:
        if letter is not None and not folder.lower().startswith(letter.lower()):
            continue

        folder_path = os.path.join(root_directory, folder)
        folder_paths.append(folder_path)

        files = os.listdir(folder_path)
        for file in files:
            img_path = os.path.join(folder_path, file)
            img_paths.append(img_path)


    if not folder_paths or not img_paths:
        raise ValueError("No folder or image path was found. Check the specified letter or path.")

    return folder_paths, img_paths