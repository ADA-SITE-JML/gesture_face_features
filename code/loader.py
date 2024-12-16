import tensorflow as tf
import os
from participant import *


model_configs = {
  'resnet50': {
    'last_conv_layer_name': 'conv5_block3_out',
    'fc_gap_layer_name': 'avg_pool',
    'input_dim': (224, 224),
  },
  'vgg19': {
    'last_conv_layer_name': 'block5_conv4',
    'fc_gap_layer_name': 'fc2',
    'input_dim': (224, 224),
  },
  'inception_v3': {
    'last_conv_layer_name': 'mixed10',
    'fc_gap_layer_name': 'avg_pool',
    'input_dim': (299, 299),
  },
  'efficientnetb0': {
    'last_conv_layer_name': 'top_conv',
    'fc_gap_layer_name': 'avg_pool',
    'input_dim': (224, 224),
  },
  'efficientnetb1': {
    'last_conv_layer_name': 'top_conv',
    'fc_gap_layer_name': 'avg_pool',
    'input_dim': (240, 240)
  },
  'efficientnetb6': {
    'last_conv_layer_name': 'top_conv',
    'fc_gap_layer_name': 'avg_pool',
    'input_dim': (528, 528),
  },
  # 'efficientnetb7': {
  #   'last_conv_layer_name': 'top_conv',
  #   'fc_gap_layer_name': 'avg_pool',
  #   'input_dim': (600, 600),
  # }
}


class Loader:
  def __init__(self, model_configs, data_path):
      self.available_models = ["vgg19", "resnet50", "inception_v3",
                                "efficientnetb0", "efficientnetb1",
                                "efficientnetb6", #"efficientnetb7",
                              ]          
      self.data_path = data_path
      self.model_configs = model_configs

      self.dataset = None
      self.models = {}
      self.img_ids = []
      self.labels = []

  def load_model(self, model_name, include_top=True):
      if model_name == 'resnet50':
          from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
          model = ResNet50(include_top=include_top)
          configs = self.model_configs[model_name]
      elif model_name == 'vgg19':
          from tensorflow.keras.applications.vgg19 import VGG19, preprocess_input
          model = VGG19(include_top=include_top)
          configs = self.model_configs[model_name]
      elif model_name == 'inception_v3':
          from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input
          model = InceptionV3(include_top=include_top)
          configs = self.model_configs[model_name]
      elif model_name == 'efficientnetb0':
          from tensorflow.keras.applications.efficientnet import EfficientNetB0, preprocess_input
          model = EfficientNetB0(include_top=include_top)
          configs = self.model_configs[model_name]
      elif model_name == 'efficientnetb1':
          from tensorflow.keras.applications.efficientnet import EfficientNetB1, preprocess_input
          model = EfficientNetB1(include_top=include_top)
          configs = self.model_configs[model_name]
      elif model_name == 'efficientnetb6':
          from tensorflow.keras.applications.efficientnet import EfficientNetB6, preprocess_input
          model = EfficientNetB6(include_top=include_top)
          configs = self.model_configs[model_name]
      else:
          raise ValueError(f"Unsupported model name. Available models are {self.available_models}")

      print(f"{model.name} loaded")
      layer_names = (configs['last_conv_layer_name'], configs['fc_gap_layer_name'])
      
      return model, layer_names, preprocess_input

  def load_dataset(self, image_size=(600, 600), interpolation="lanczos5"):
      self.labels = [os.path.basename(root) for root, _, files in os.walk(self.data_path) for _ in files]
      img_names = [f for _, _, files in os.walk(self.data_path) for f in files]
      self.img_ids = [int(f.split('_')[1].replace('.JPG', '')) for f in img_names]
      img_ids_dataset = tf.data.Dataset.from_tensor_slices(self.img_ids)

      self.dataset = tf.keras.utils.image_dataset_from_directory(
          directory=self.data_path,
          batch_size=None,
          image_size=image_size,
          shuffle=False,
          interpolation=interpolation,
      )

      self.dataset = tf.data.Dataset.zip((self.dataset, img_ids_dataset))
      self.dataset = self.dataset.map(lambda imgs_labels, img_ids: (imgs_labels[0], imgs_labels[1], img_ids))

  def load_models(self):
      for model_name in self.available_models:
        self.models[model_name] = self.load_model(model_name)