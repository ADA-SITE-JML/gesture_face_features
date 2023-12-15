import cv2
import mediapipe as mp
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import os

def extract_hand_md(images, input_dim=None):
    if not isinstance(images, list):
        images = [images]

    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands()

    for img in images:
        if not input_dim:
            img_resized = img.copy()
        else:
            img_resized = img.resize(input_dim)
        img_rgb = cv2.cvtColor(np.array(img_resized), cv2.COLOR_RGB2BGR)
        results = hands.process(img_rgb)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                h, w, _ = np.array(img_resized).shape
                hand_points = np.array([[lm.x * w, lm.y * h] for lm in hand_landmarks.landmark], dtype=np.int32)

                hand_img = get_cropped_hand(img_resized, hand_points)

                plt.imshow(hand_img)
                plt.title('Cropped Hand Image')
                plt.axis('off')
                plt.show()

        else:
            image_filename = os.path.splitext(os.path.basename(img.filename))[0]
            print(f'No hand detected in the image {image_filename}.')


def get_cropped_hand(img, hand_landmarks):
    x_min, y_min, x_max, y_max = get_hand_bbox(hand_landmarks)

    cropped_hand = np.array(img.copy())
    cropped_hand = cropped_hand[y_min:y_max, x_min:x_max]

    return cropped_hand


def get_hand_bbox(hand_landmarks):
    x_values = hand_landmarks[:, 0]
    y_values = hand_landmarks[:, 1]

    x_min, x_max = np.min(x_values), np.max(x_values)
    y_min, y_max = np.min(y_values), np.max(y_values)

    return int(x_min), int(y_min), int(x_max), int(y_max)