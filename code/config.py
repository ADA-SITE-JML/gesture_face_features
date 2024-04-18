from numpy.core.fromnumeric import nonzero
from utils import load_imgs, get_paths
import os
import sys
from keras.models import Model
import numpy as np


# TODO: Squeezenet doesn't return preprocess_input and 
# decode_predictions yet
# TODO: VGG19 shape gives error when 
# calling get_heatmaps (see heatmap.py)
available_models = [
  "VGG19",
  "ResNet50",
  "InceptionV3",
  "EfficientNetB0",
  "EfficientNetB1",
  "EfficientNetB6",
  "squeezenet",
]

available_letters = ["A", "H", "L", "N", "O", "P", "R"]

participants = {
  0: list(range(2914, 2951)),
  1: list(range(2871, 2904)),
  2: list(range(2323, 2356)),
  3: list(range(2285, 2314)),
  4: list(range(1646, 1675)),
  5: list(range(1510 , 1544)),
}

test_imgs = [2919, 2871, 2325, 2287, 1646, 1539]


class Config:
    def __init__(self, 
                  path, 
                  img_type="sign",
                  load_data=False, 
                  shuffle=False, 
                  letter=None, 
                  is_random=False,
                  img_numbers = None,
                  model_name="InceptionV3", 
                  debug=False,
                  # Default for ResNet, VGG19, squeezenet
                  input_size=224,
                  input_dim=(224, 224),
                  feats = 0, 
                  feat_layer = 0,
                  last_layer_weights = 0,
                  preprocess_input = None
                ):

        self.path = path
        self.img_type = img_type
        self.load_data = load_data
        self.shuffle = shuffle
        self.letter = letter
        self.img_numbers = img_numbers
        self.is_random = is_random
        
        self.model_name = model_name
        self.debug = debug
        self.input_size = input_size
        self.input_dim = input_dim
        self.feats = feats
        self.feat_layer = feat_layer
        self.last_layer_weights = last_layer_weights

        self.model = None
        self.transfer_model = None

        self.code_path = os.path.join(self.path, "code")
        self.samples_path = os.path.join(self.path, "samples")
        self.pillow_path = os.path.join(self.path, "pillow")
        self.heatmaps_path = os.path.join(self.path, "heatmaps", self.model_name)
        self.data_path = os.path.join(self.samples_path, self.img_type)

        self.folder_paths = []
        self.img_paths = []
        self.imgs = []

        self.folder_paths, self.img_paths = get_paths(self.data_path, self.letter, self.is_random, self.img_numbers)
        
        if load_data:
            print("Loading data...")         
            self.imgs = load_imgs(self.img_paths, self.shuffle, self.letter)
            print("Data loaded successfully.")

        self.img_count = len(self.imgs)

        print(f'Setting up model {self.model_name}...')

        if self.model_name == 'ResNet50':
          from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
          self.feats = 2048
          self.model = ResNet50(weights='imagenet')
          self.feat_layer = -3
        elif self.model_name == 'VGG19':
          from tensorflow.keras.applications.vgg19 import VGG19, preprocess_input
          self.input_size = 224
          self.input_dim = (224,224)
          self.feats = 512
          self.model = VGG19(weights='imagenet')
          self.feat_layer = -6
        elif self.model_name == 'InceptionV3':
          from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input
          self.input_size = 296
          self.input_dim = (299,299)
          self.feats = 2048
          self.model = InceptionV3(weights='imagenet')
          self.feat_layer = -3
        elif self.model_name == 'EfficientNetB0':
          from tensorflow.keras.applications.efficientnet import EfficientNetB0, preprocess_input
          self.feats = 1280
          self.model = EfficientNetB0(weights='imagenet')
          self.feat_layer = -4
        elif self.model_name == 'EfficientNetB1':
          from tensorflow.keras.applications.efficientnet import EfficientNetB1, preprocess_input
          self.input_size = 240
          self.input_dim = (240,240)
          self.feats = 1280
          self.model = EfficientNetB1(weights='imagenet')
          self.feat_layer = -4
        elif self.model_name == 'EfficientNetB6':
          from tensorflow.keras.applications.efficientnet import EfficientNetB6, preprocess_input
          self.input_size = 527
          self.input_dim = (528,528)
          self.feats = 2304
          self.model = EfficientNetB6(weights='imagenet')
          self.feat_layer = -4
        elif self.model_name == 'squeezenet':
          import torchvision
          from torchvision.models import squeezenet1_1
          from torchvision.models.feature_extraction import create_feature_extractor
          self.feats = 86528
          self.model = squeezenet1_1(pretrained=True)
          return_nodes = { 'features.12.cat': 'layer12' }
          self.model = create_feature_extractor(self.model, return_nodes=return_nodes)
        else:
          print('Model is not supported!')
          sys.exit()
        
        if debug:
          print(self.model.summary()) #Notice the Global Average Pooling layer at the last but onemodel = VGG19(weights='imagenet')

        # For keras models: Get weights for the prediction layer (last layer)
        # Output both predictions (last layer) and conv5_block3_add (just before final activation layer)
        if self.model_name != 'squeezenet':
          self.preprocess_input = preprocess_input
          self.last_layer_weights = self.model.layers[-1].get_weights()[0]  #Predictions layer
          self.transfer_model = Model(inputs=self.model.input, outputs=(self.model.layers[self.feat_layer].output, self.model.layers[-1].output))
          
        print(f'Model was set up successfully.')

        self.heatmap_config = [
          self.model_name,
          self.preprocess_input,  
          self.transfer_model, 
          self.last_layer_weights, 
          self.input_size, 
          self.feats,
          self.input_dim,
          self.imgs,
          self.img_paths,
          self.heatmaps_path,
        ]


    def get_last_conv_outputs(self):
        all_last_conv_outputs = []

        for img in self.imgs:
            img_resized = img.resize(self.input_dim)
            img_tensor = np.expand_dims(np.array(img_resized), axis=0)
            preprocessed_img = self.preprocess_input(img_tensor)

            last_conv_output, _ = self.transfer_model.predict(preprocessed_img)
            last_conv_output_flat = np.squeeze(last_conv_output).reshape((last_conv_output.shape[0], -1))

            all_last_conv_outputs.append(last_conv_output_flat)

        return np.vstack(all_last_conv_outputs)


    def __repr__(self):
        attributes = [
            f"img_type={self.img_type}",
            f"model_name={self.model_name}",
            f"load_data={self.load_data}",
            f"shuffle={self.shuffle}",
            f"img_count={self.img_count}",
            f"letter={self.letter}",
            f"input_size={self.input_size}",
            f"input_dim={self.input_dim}",
            f"feats={self.feats}",
            f"feat_layer={self.feat_layer}",
        ]
        return "Config(\n  " + ",\n  ".join(attributes) + "\n)"
       