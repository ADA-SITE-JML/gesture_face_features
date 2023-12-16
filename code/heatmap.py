# Source: https://github.com/bnsreenu/python_for_microscopists/blob/
# master/263_Object%20localization%20in%20images%E2%80%8B
# _using_GAP_layer/263_Object%20localization%20in%20images%E2%80%8B_using_GAP_layer.py
import numpy as np
import os
import scipy
import matplotlib.pyplot as plt
from PIL import Image

class Heatmap:
    def __init__(self, preprocess_input, decode_predictions, transfer_model, last_layer_weights, input_size, feats, input_dim, imgs, img_paths, heatmap_path):
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

    def get_heatmap(self, input_img, img_type):
        img_tensor = np.expand_dims(input_img, axis=0)
        preprocessed_img = self.preprocess_input(img_tensor)

        last_conv_output, pred_vec = self.transfer_model.predict(preprocessed_img)

        last_conv_output = np.squeeze(last_conv_output)
        pred = np.argmax(pred_vec)

        h = int(input_img.shape[0] / last_conv_output.shape[0])
        w = int(input_img.shape[1] / last_conv_output.shape[1])

        upsampled_last_conv_output = scipy.ndimage.zoom(last_conv_output, (h, w, 1), order=1)
        last_layer_weights_for_pred = self.last_layer_weights[:, pred]

        heat_map = np.dot(upsampled_last_conv_output.reshape((self.input_size * self.input_size, self.feats)),
                          last_layer_weights_for_pred).reshape(self.input_size, self.input_size)

        return heat_map


    def generate_heatmaps(self, num_images=None):
        for img_type, img_list in self.imgs.items():
            img_paths = self.img_paths[img_type]

            if num_images is not None:
                img_list = img_list[:num_images]
                img_paths = img_paths[:num_images]

            for i, (img, img_path) in enumerate(zip(img_list, img_paths)):
                img_array = np.array(img.resize(self.input_dim))
                heatmap = self.get_heatmap(img_array, img_type)

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

                if self.heatmap_path:
                    os.makedirs(self.heatmap_path, exist_ok=True)
                    image_filename = os.path.splitext(os.path.basename(img_path))[0]
                    category_folder = os.path.join(self.heatmap_path, img_type.lower())
                    os.makedirs(category_folder, exist_ok=True)

                    heatmap_filename = f"{image_filename}_heatmap.JPG"
                    heatmap_filepath = os.path.join(category_folder, heatmap_filename)

                    if not os.path.exists(heatmap_filepath):
                        fig.savefig(heatmap_filepath)
                        print(f"Heatmap saved: {heatmap_filepath}")

                plt.show()


    def load_heatmaps(self, num_images=None):
        if not isinstance(self.img_paths, dict):
            print("Invalid input. 'img_paths' should be a dictionary.")
            return

        for img_type, img_path_list in self.img_paths.items():
            category_folder = os.path.join(self.heatmap_path, img_type.lower())
            os.makedirs(category_folder, exist_ok=True)

            if num_images is not None:
                img_path_list = img_path_list[:num_images]

            for i, img_path in enumerate(img_path_list):
                if not os.path.exists(img_path):
                    print(f"Invalid input. Image file not found: {img_path}")
                    continue

                image_name = os.path.splitext(os.path.basename(img_path))[0]
                heatmap_filename = f"{image_name}_heatmap.JPG"
                heatmap_filepath = os.path.join(category_folder, heatmap_filename)

                if os.path.exists(heatmap_filepath):
                    fig = plt.figure()
                    heatmap_image = plt.imread(heatmap_filepath)
                    plt.imshow(heatmap_image)
                    plt.title(f'Heatmap for {image_name}')
                    plt.show()
                else:
                    print(f"Heatmap not found for {image_name}")