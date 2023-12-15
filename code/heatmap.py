# Source: https://github.com/bnsreenu/python_for_microscopists/blob/
# master/263_Object%20localization%20in%20images%E2%80%8B
# _using_GAP_layer/263_Object%20localization%20in%20images%E2%80%8B_using_GAP_layer.py

import numpy as np
from google.colab.patches import cv2_imshow
import matplotlib.pyplot as plt
import os
import csv
import scipy #to upscale the image
from PIL import Image
import cv2

def get_heatmap(input_img, 
                preprocess_input, 
                decode_predictions, 
                transfer_model, 
                last_layer_weights, 
                input_size, 
                feats
              ):
    
    
    img_tensor = np.expand_dims(input_img, axis=0)

    preprocessed_img = preprocess_input(img_tensor)

    #Get the predictions and the output of last conv. layer.
    last_conv_output, pred_vec = transfer_model.predict(preprocessed_img)

    print('last_conv_output',last_conv_output.shape)
    #Last conv. output for the image
    last_conv_output = np.squeeze(last_conv_output) # usually 7x7xfeats
    #Prediction for the image
    pred = np.argmax(pred_vec)

    decode_predictions(pred_vec, top=1)

    # spline interpolation to resize each filtered image to size of original image
    h = int(input_img.shape[0]/last_conv_output.shape[0])
    w = int(input_img.shape[1]/last_conv_output.shape[1])

    upsampled_last_conv_output = scipy.ndimage.zoom(last_conv_output, (h, w, 1), order=1) # dim: imput_size x imput_size x feats
    print('upsampled_last_conv_output',upsampled_last_conv_output.shape)

    #Get the weights from the last layer for the prediction class
    last_layer_weights_for_pred = last_layer_weights[:, pred] # dim: (2048,)
    print('last_layer_weights_for_pred',last_layer_weights_for_pred.shape)

    # To generate the final heat map.
    # Reshape the upsampled last conv. output to n x filters and multiply (dot product)
    # with the last layer weigths for the prediction.
    # Reshape back to the image size for easy overlay onto the original image.
    heat_map = np.dot(upsampled_last_conv_output.reshape((input_size*input_size, feats)),
                  last_layer_weights_for_pred).reshape(input_size,input_size)
    return heat_map


def generate_heatmaps(images, input_dim, preprocess_input, decode_predictions, transfer_model, last_layer_weights, input_size, feats, heatmaps_path):
    if not isinstance(images, list):
        images = [images]

    for img in images:
        img_resized = img.resize(input_dim)
        img_array = np.array(img_resized)

        heatmap = get_heatmap(img_array, preprocess_input, decode_predictions, transfer_model, last_layer_weights, input_size, feats)

        fig, axs = plt.subplots(1, 2, figsize=(12, 6))

        # Plot the original image in the first subplot
        axs[0].imshow(img_resized)
        axs[0].set_title('Original Image')
        axs[0].axis('off')

        # Plot the heatmap drawn on top of the original image in the second subplot
        axs[1].imshow(img_resized)
        axs[1].imshow(heatmap, cmap='jet', alpha=0.5)
        axs[1].set_title('Heatmap')
        axs[1].axis('off')

        if heatmaps_path:
            os.makedirs(heatmaps_path, exist_ok=True)
            image_filename = os.path.splitext(os.path.basename(img.filename))[0]
            heatmap_filename = f"{image_filename}_heatmap.JPG"
            heatmap_filepath = os.path.join(heatmaps_path, heatmap_filename)

            if not os.path.exists(heatmap_filepath):
                fig.savefig(heatmap_filepath)
                print(f"Heatmap saved: {heatmap_filepath}")

        plt.show()


def load_heatmaps(images, heatmaps_path):
    if not isinstance(images, list):
        images = [images]

    for img in images:
        if isinstance(img, Image.Image):
            image_name = os.path.splitext(os.path.basename(img.filename))[0]
        else:
            print("Invalid input. Please provide a list of PIL images or a single PIL image.")
            return

        heatmap_filename = f"{image_name}_heatmap.JPG"
        heatmap_filepath = os.path.join(heatmaps_path, heatmap_filename)

        if os.path.exists(heatmap_filepath):
            fig = plt.figure()
            fig = plt.imread(heatmap_filepath)
            plt.imshow(fig)
            plt.title(f'Heatmap for {image_name}')
            plt.show()
        else:
            print(f"Heatmap not found for {image_name}")
