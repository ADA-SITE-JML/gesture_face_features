# utils.py
import os
from PIL import Image, ImageEnhance
import random
import matplotlib.pyplot as plt
import numpy as np
from google.colab.patches import cv2_imshow
import random


def adjust_contrast(img, factor):
    enhancer = ImageEnhance.Contrast(img)
    enhanced_img = enhancer.enhance(factor)
    return enhanced_img

def adjust_contrasts(img_list, factor):
    enhanced_imgs = []
    for img in img_list:
        enhanced_imgs.append(adjust_contrast(img, factor))
    return enhanced_imgs


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


def load_imgs(img_paths, shuffle=False, letter=None):
    if shuffle:
        random.shuffle(img_paths)

    loaded_imgs = []
    for path in img_paths:
        # Extract the last part of the path (letter)
        folder_name = os.path.basename(os.path.dirname(path)).lower()

        if letter is not None and folder_name not in [l.lower() for l in letter]:
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


def get_paths(root_directory, letter=None, is_random=False, img_numbers=None):
    folder_paths = []
    img_paths = []

    folders = os.listdir(root_directory)

    if letter is not None:
        if isinstance(letter, str):
            letter = [letter.upper()]
        else:
            letter = [l.upper() for l in letter]
        folders = [folder for folder in folders if folder.upper().startswith(tuple(letter))]

    if not folders:
        print("No folders found matching the criteria.")
        return folder_paths, img_paths  # Return empty lists if no folders match

    for folder in folders:
        folder_path = os.path.join(root_directory, folder)
        folder_paths.append(folder_path)

        files = os.listdir(folder_path)
        if img_numbers is not None:
            # Build filenames from img_numbers and check if they exist
            img_files = [f"IMG_{num}.JPG" for num in img_numbers]
            found_images = [f for f in img_files if f in files]
            if not found_images:
                print(f"No images found for the numbers provided in folder {folder}.")
            img_paths.extend(os.path.join(folder_path, img) for img in found_images)
        elif is_random:
            if files:
                random_img = random.choice(files)
                img_paths.append(os.path.join(folder_path, random_img))
            else:
                print(f"No files to select randomly in folder {folder}.")
        else:
            img_paths.extend(os.path.join(folder_path, f) for f in files)

    return folder_paths, img_paths