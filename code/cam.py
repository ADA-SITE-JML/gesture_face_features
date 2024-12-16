import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

class Gradcam:
    def __init__(self, model, layer_name, input_dim=(224, 224)):
        """
        Initialize the GradCam class.

        Parameters:
        - model (tf.keras.Model): The pre-trained Keras model for prediction.
        - layer_name (str): Name of the layer of the model for Grad-CAM.
        - input_dim (tuple): Target dimensions for resizing the image (height, width).
        """
        self.model = model
        self.layer_name = layer_name
        self.input_dim = input_dim

    def get_gradcam_heatmap(self, img_array, pred_index=None):
        """
        Generate the Grad-CAM heatmap for a given image.

        Parameters:
        - img_array (tensorflow tensor): The image tensor to process.
        - pred_index (int, optional): Index of the class to visualize. If None, the predicted class is used.

        Returns:
        - heatmap (numpy array): The generated Grad-CAM heatmap.
        """
        self.model.layers[-1].activation = None
        grad_model = tf.keras.models.Model(
            self.model.inputs, [self.model.get_layer(self.layer_name).output, self.model.output]
        )

        with tf.GradientTape() as tape:
            layer_output, preds = grad_model(img_array)
            if pred_index is None:
                pred_index = tf.argmax(preds[0])
            class_channel = preds[:, pred_index]

        grads = tape.gradient(class_channel, layer_output)
        
        # Check the shape of grads to apply the correct reduction axes
        grads_shape = grads.shape
        if len(grads_shape) > 3:  # If grads is a 4D tensor (height, width, channels, batch)
            pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        elif len(grads_shape) == 3:  # If grads is 3D (channels, height, width)
            pooled_grads = tf.reduce_mean(grads, axis=(0, 1))
        else:  # If grads is 2D (batch, channels), this is rare but needs handling
            pooled_grads = tf.reduce_mean(grads, axis=0)

        layer_output = layer_output[0]
        
        # Ensure the shape of pooled_grads is broadcastable to layer_output
        pooled_grads = tf.reshape(pooled_grads, [1, 1, 1, -1])  # Reshape to (1, 1, 1, channels)

        # Now we can perform element-wise multiplication
        heatmap = layer_output * pooled_grads
        heatmap = tf.reduce_sum(heatmap, axis=-1)  # Sum across the channels

        # Apply ReLU and normalize the heatmap
        heatmap = tf.maximum(heatmap, 0)
        heatmap = heatmap / tf.reduce_max(heatmap)

        return heatmap.numpy()

    def superimpose(self, img_array, heatmap, alpha=0.6):
        """
        Superimpose the Grad-CAM heatmap on the image tensor.

        Parameters:
        - img_array (tensorflow tensor): The image tensor to process.
        - heatmap (numpy array): The Grad-CAM heatmap.
        - alpha (float): The transparency of the heatmap.

        Returns:
        - superimposed_img (tensorflow tensor): The image with the superimposed heatmap.
        """
        img_array = (img_array - tf.reduce_min(img_array)) / (tf.reduce_max(img_array) - tf.reduce_min(img_array)) * 255
        heatmap = np.uint8(255 * heatmap)

        jet = mpl.colormaps["jet"]
        jet_colors = jet(np.arange(256))[:, :3]
        jet_heatmap = jet_colors[heatmap]

        jet_heatmap = tf.image.resize(jet_heatmap, (img_array.shape[0], img_array.shape[1]))
        jet_heatmap = tf.convert_to_tensor(jet_heatmap, dtype=tf.float32)

        superimposed_img = jet_heatmap * alpha + img_array
        return superimposed_img

    def gradcam(self, img, pred_index=None, verbose=True, alpha=0.6):
        """
        Process the image, predict using the model, generate a Grad-CAM heatmap,
        and superimpose the heatmap on the original image.

        Parameters:
        - img (tensorflow tensor): The input image tensor.
        - pred_index (int, optional): Index of the class to visualize. If None, the predicted class is used.
        - verbose (bool): If True, prints the model prediction.
        - alpha (float): Transparency of the heatmap.

        Returns:
        - heatmap (numpy array): The Grad-CAM heatmap.
        - superimposed (tensorflow tensor): The superimposed image.
        """
        # Resize the image tensor
        img_resized = tf.image.resize(img, self.input_dim)

        img_array = tf.expand_dims(img_resized, axis=0)
        
        # If you have a custom model, you may not need preprocessing like VGG or ResNet
        preds = self.model.predict(img_array)
        heatmap = self.get_gradcam_heatmap(img_array, pred_index)
        superimposed = self.superimpose(img_resized, heatmap, alpha=alpha)

        if verbose:
            # Optionally print model predictions if necessary
            print("Predicted:", preds)
            self.plot_img_grid([heatmap, superimposed.numpy()], 2, ["heatmap", "superimposed image"])

        return heatmap, superimposed

    def plot_img_grid(self, images, n_cols, titles=None):
        """
        Plot a grid of images.

        Parameters:
        - images (list): List of images to plot.
        - n_cols (int): Number of columns in the grid.
        - titles (list, optional): Titles for each image.
        """
        n_rows = (len(images) + n_cols - 1) // n_cols
        plt.figure(figsize=(n_cols * 5, n_rows * 5))

        for i, img in enumerate(images):
            plt.subplot(n_rows, n_cols, i + 1)
            plt.imshow(img)
            if titles:
                plt.title(titles[i])
            plt.axis('off')

        plt.tight_layout()
        plt.show()

    def plot_gradcams(self, img_ids, imgs, verbose=False):
        """
        Plot Grad-CAM results for a list of images.

        Parameters:
        - img_ids (list): List of image IDs to process.
        - imgs (list): List of images to process.
        - verbose (bool): If True, prints the model prediction for each image.
        """
        superimposed_imgs = []

        for img_id in img_ids:
            img = imgs[img_id]
            _, superimposed = self.gradcam(img, verbose=verbose)
            superimposed_imgs.append(superimposed)

        plt.figure(figsize=(15, 5))

        plt.subplot(1, len(superimposed_imgs) + 1, 1)  # Extra space for the title
        plt.text(0.5, 0.5, self.model.name, fontsize=16, ha='center')  # Centered title
        plt.axis('off')

        for i, heatmap in enumerate(superimposed_imgs):
            plt.subplot(1, len(superimposed_imgs) + 1, i + 2)
            plt.imshow(heatmap)
            plt.axis('off')

        plt.tight_layout()
        plt.show()
