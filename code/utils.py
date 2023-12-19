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


def show_img(img, ax, title=None):
    ax.imshow(np.array(img))
    if title:
        ax.set_title(title)
    ax.axis('off')



def show_img(img, ax, title=None):
    ax.imshow(np.array(img))
    if title:
        ax.set_title(title)
    ax.axis('off')

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



def get_paths(root_directory, letter=None, is_random=False):
    folder_paths = []
    img_paths = []

    folders = os.listdir(root_directory)
    for folder in folders:
        if letter is not None and not folder.lower().startswith(letter.lower()):
            continue

        folder_path = os.path.join(root_directory, folder)
        folder_paths.append(folder_path)

        files = os.listdir(folder_path)
        if is_random:
            random_img = random.choice(files)
            img_path = os.path.join(folder_path, random_img)
            img_paths.append(img_path)
        else:
            for f in files:
                img_path = os.path.join(folder_path, f)
                img_paths.append(img_path)

    if not folder_paths or not img_paths:
        raise ValueError("No folder or image path was found. Check the specified letter or path.")

    return folder_paths, img_paths