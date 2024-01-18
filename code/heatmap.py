# Source: https://github.com/bnsreenu/python_for_microscopists/blob/
# master/263_Object%20localization%20in%20images%E2%80%8B
# _using_GAP_layer/263_Object%20localization%20in%20images%E2%80%8B_using_GAP_layer.py
import numpy as np
import os
import scipy
import matplotlib.pyplot as plt
from PIL import Image

class Heatmap:
    def __init__(self, 
                  model_name,
                  preprocess_input, 
                  decode_predictions, 
                  transfer_model, 
                  last_layer_weights, 
                  input_size, 
                  feats, 
                  input_dim, 
                  imgs, 
                  img_paths, 
                  heatmap_path
                ):

        self.model_name = model_name
        self.preprocess_input = preprocess_input
        self.decode_predictions = decode_predictions
        self.transfer_model = transfer_model
        self.last_layer_weights = last_layer_weights
        self.input_size = input_size
        self.feats = feats
        self.input_dim = input_dim
        self.imgs = imgs
        self.img_paths = img_paths
        self.heatmap_path = heatmap_path

    def get_heatmap(self, input_img, mean=True):
        img_tensor = np.expand_dims(input_img, axis=0)
        preprocessed_img = self.preprocess_input(img_tensor)

        last_conv_output, pred_vec = self.transfer_model.predict(preprocessed_img)

        print('Input Image Shape:', input_img.shape)
        print('Preprocessed Image Shape:', preprocessed_img.shape)
        print('Last Conv Output Shape:', last_conv_output.shape)

        last_conv_output = np.squeeze(last_conv_output)
        pred = np.argmax(pred_vec)

        print('Squeezed Last Conv Output Shape:', last_conv_output.shape)
        print('Predicted Class:', pred)

        h = int(input_img.shape[0] / last_conv_output.shape[0])
        w = int(input_img.shape[1] / last_conv_output.shape[1])

        upsampled_last_conv_output = scipy.ndimage.zoom(last_conv_output, (h, w, 1), order=1)
        
        print('Upsampled Last Conv Output Shape:', upsampled_last_conv_output.shape)

        last_layer_weights_for_pred = self.last_layer_weights[:, pred]

        print('Last Layer Weights for Predicted Class Shape:', last_layer_weights_for_pred.shape)

        reshaped_array = upsampled_last_conv_output.reshape((self.input_size * self.input_size, self.feats))

        print("Reshaped Array", reshaped_array.shape)

        
        if self.model_name == "VGG19":
          if mean:
            reshaped_array = np.mean(reshaped_array, axis=1, keepdims=True)
          else:
            reshaped_array = np.amax(reshaped_array, axis=1, keepdims=True)
          reshaped_array = np.tile(reshaped_array, last_layer_weights_for_pred.shape[0])
          print("VGG19 depth averaged or maxed:", reshaped_array.shape)
        
        heat_map = np.dot(reshaped_array, last_layer_weights_for_pred).reshape(self.input_size, self.input_size)

        print('Final Heat Map Shape:', heat_map.shape)

        return heat_map



    def generate_heatmaps(self, mean=True, img_count=None, save=False):
        for img_type, img_list in self.imgs.items():
            img_paths = self.img_paths[img_type]

            if img_count is not None:
                img_list = img_list[:img_count]
                img_paths = img_paths[:img_count]

            for i, (img, img_path) in enumerate(zip(img_list, img_paths)):
                img_array = np.array(img.resize(self.input_dim))
                heatmap = self.get_heatmap(img_array, mean=mean)

                fig, axs = plt.subplots(1, 2, figsize=(12, 6))

                # Plot the original image in the first subplot
                axs[0].imshow(img_array)
                axs[0].set_title('Original Image')
                axs[0].axis('off')

                # Plot the heatmap drawn on top of the original image in the second subplot
                axs[1].imshow(img_array)
                axs[1].imshow(heatmap, cmap='jet', alpha=0.5)
                axs[1].set_title('Heatmap')
                axs[1].axis('off')

                if save:
                    os.makedirs(self.heatmap_path, exist_ok=True)
                    img_filename = os.path.splitext(os.path.basename(img_path))[0]
                    letter = img_path.split("/")[-2]
                    category_folder = os.path.join(self.heatmap_path, letter, img_type.lower())
                    os.makedirs(category_folder, exist_ok=True)

                    heatmap_filename = f"{img_filename}_heatmap.JPG"
                    heatmap_filepath = os.path.join(category_folder, heatmap_filename)

                    if not os.path.exists(heatmap_filepath):
                        fig.savefig(heatmap_filepath)
                        print(f"Heatmap saved: {heatmap_filepath}")

                plt.show()


    def generate_heatmaps_row(self, mean=True, img_count=None, imgs_per_row=5):
        for img_type in ["sign", "face"]:
            img_list = self.imgs[img_type]
            img_paths = self.img_paths[img_type]
            if img_count is not None:
                img_list = img_list[:img_count]
                img_paths = img_paths[:img_count]

            num_rows = (len(img_list) + imgs_per_row - 1) // imgs_per_row
            for row in range(num_rows):
                fig, axs = plt.subplots(1, imgs_per_row, figsize=(4 * imgs_per_row, 6))
                for i in range(imgs_per_row):
                    idx = row * imgs_per_row + i
                    if idx < len(img_list):
                        img, img_path = img_list[idx], img_paths[idx]
                        img_array = np.array(img.resize(self.input_dim))
                        heatmap = self.get_heatmap(img_array, mean=mean)

                        axs[i].imshow(img_array)
                        axs[i].imshow(heatmap, cmap='jet', alpha=0.5)
                        axs[i].axis('off')

                plt.show()


    def load_heatmaps(self, img_count=None):
        if not isinstance(self.img_paths, dict):
            print("Invalid input. 'img_paths' should be a dictionary.")
            return

        for img_type, img_path_list in self.img_paths.items():
            if img_count is not None:
                img_path_list = img_path_list[:img_count]

            for i, img_path in enumerate(img_path_list):
                if not os.path.exists(img_path):
                    print(f"Invalid input. Image file not found: {img_path}")
                    continue

                img_name = os.path.splitext(os.path.basename(img_path))[0]
                letter = img_path.split("/")[-2]
                heatmap_filename = f"{img_name}_heatmap.JPG"
                heatmap_filepath = os.path.join(self.heatmap_path, letter, img_type.lower(), heatmap_filename)
                print(heatmap_filepath)
                
                if os.path.exists(heatmap_filepath):
                    fig = plt.figure()
                    heatmap_img = plt.imread(heatmap_filepath)
                    plt.imshow(heatmap_img)
                    plt.title(f'Heatmap for {img_name}')
                    plt.show()
                else:
                    print(f"Heatmap not found for {img_name}")