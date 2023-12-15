from numpy.core.fromnumeric import nonzero
from utils import load_imgs, get_paths
import os
import sys
from keras.models import Model

class Config:
    def __init__(self, 
                  path, 
                  img_type="both",
                  load_data=True, 
                  shuffle=False, 
                  letter=None, 
                  model_name="InceptionV3", 
                  debug=False,
                  # Default for ResNet, VGG19, squeezenet
                  input_size=224,
                  input_dim=(224, 224),
                  feats = 0, 
                  feat_layer = 0,
                  last_layer_weights = 0
                ):

        self.path = path
        self.img_type = img_type
        self.load_data = load_data
        self.shuffle = shuffle
        self.letter = letter

        self.code_path = os.path.join(self.path, "code")
        self.data_path = os.path.join(self.path, "samples")
        self.pillow_path = os.path.join(self.path, "pillow")
        self.heatmaps_path = os.path.join(self.path, "heatmaps")
        self.sign_path = os.path.join(self.data_path, "sign")
        self.face_path = os.path.join(self.data_path, "face")

        self.sign_folder_paths, self.sign_img_paths = get_paths(self.sign_path, letter=self.letter)
        self.face_folder_paths, self.face_img_paths = get_paths(self.face_path, letter=self.letter)

        self.sign_imgs = []
        self.face_imgs = []

        self.model_name = model_name
        self.debug = debug
        self.input_size = input_size
        self.input_dim = input_dim
        self.feats = feats
        self.feat_layer = feat_layer
        self.last_layer_weights = last_layer_weights
        self.model = None
        self.transfer_model = None

        if load_data:
            print("Loading data...")
            if self.img_type == 'sign':
                self.sign_imgs = load_imgs(self.sign_img_paths, shuffle=self.shuffle)
            elif self.img_type == 'face':
                self.face_imgs = load_imgs(self.face_img_paths, shuffle=self.shuffle)
            elif self.img_type == 'both':
                self.sign_imgs = load_imgs(self.sign_img_paths, shuffle=self.shuffle)
                self.face_imgs = load_imgs(self.face_img_paths, shuffle=self.shuffle)
            print("Data loaded successfully.")

        self.img_count = len(self.sign_imgs) + len(self.face_imgs)


        print(f'Setting up model {self.model_name}...')

        if self.model_name == 'ResNet50':
          from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input, decode_predictions
        elif self.model_name == 'VGG19':
          from tensorflow.keras.applications.vgg19 import VGG19, preprocess_input, decode_predictions
        elif self.model_name == 'InceptionV3':
          from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input, decode_predictions
        elif self.model_name == 'EfficientNetB0':
          from tensorflow.keras.applications.efficientnet import EfficientNetB0, preprocess_input, decode_predictions
        elif self.model_name == 'EfficientNetB1':
          from tensorflow.keras.applications.efficientnet import EfficientNetB1, preprocess_input, decode_predictions
        elif self.model_name == 'EfficientNetB6':
          from tensorflow.keras.applications.efficientnet import EfficientNetB6, preprocess_input, decode_predictions
        elif self.model_name == 'squeezenet':
          import torchvision
          from torchvision.models import squeezenet1_1
          from torchvision.models.feature_extraction import create_feature_extractor
        else:
          print('Model is not supported!')
          sys.exit()

        self.preprocess_input = preprocess_input
        self.decode_predictions = decode_predictions

        if self.model_name == 'ResNet50':
          self.feats = 2048
          self.model = ResNet50(weights='imagenet')
        elif self.model_name == 'VGG19':
          self.input_size = 224
          self.input_dim = (224,224)
          self.feats = 512
          self.model = VGG19(weights='imagenet')
        elif self.model_name == 'InceptionV3':
          self.input_size = 296
          self.input_dim = (299,299)
          self.feats = 2048
          self.model = InceptionV3(weights='imagenet')
        elif self.model_name == 'EfficientNetB0':
          self.feats = 1280
          self.model = EfficientNetB0(weights='imagenet')
        elif self.model_name == 'EfficientNetB1':
          self.input_size = 240
          self.input_dim = (240,240)
          self.feats = 1280
          self.model = EfficientNetB1(weights='imagenet')
        elif self.model_name == 'EfficientNetB6':
          self.input_size = 527
          self.input_dim = (528,528)
          self.feats = 2304
          self.model = EfficientNetB6(weights='imagenet')
        elif self.model_name == 'squeezenet':
          self.feats = 86528
          self.model = squeezenet1_1(pretrained=True)
        else:
          print('Model is not supported!')
          sys.exit()

        if debug:
          print(self.model.summary()) #Notice the Global Average Pooling layer at the last but onemodel = VGG19(weights='imagenet')

        # Get weights for the prediction layer (last layer)
        if self.model_name != 'squeezenet':
          self.last_layer_weights = self.model.layers[-1].get_weights()[0]  #Predictions layer

        if self.model_name == 'ResNet50':
          self.feat_layer = -3
        elif self.model_name == 'VGG19':
          self.feat_layer = -6
        elif self.model_name == 'InceptionV3':
          self.feat_layer = -3
        elif self.model_name == 'EfficientNetB0':
          self.feat_layer = -4
        elif self.model_name == 'EfficientNetB1':
          self.feat_layer = -4
        elif model_name == 'EfficientNetB6':
          self.feat_layer = -4
        elif self.model_name == 'squeezenet':
          return_nodes = { 'features.12.cat': 'layer12' }
          self.model = create_feature_extractor(self.model, return_nodes=return_nodes)
        else:
          print('Model is not supported!')
          sys.exit()

        # For keras models: Output both predictions (last layer) and conv5_block3_add (just before final activation layer)
        if model_name != 'squeezenet':
          self.transfer_model = Model(inputs=self.model.input, outputs=(self.model.layers[self.feat_layer].output, self.model.layers[-1].output))

        print(f'Model was set up successfully.')

        self.heatmap_config =  [
          self.preprocess_input, 
          self.decode_predictions, 
          self.transfer_model, 
          self.last_layer_weights, 
          self.input_size, 
          self.feats
        ]

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
       