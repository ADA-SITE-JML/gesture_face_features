from string import ascii_letters
import matplotlib.pyplot as plt
from torchvision import transforms
from torchvision.datasets import ImageFolder
import os
from torch import device, cuda
from torchvision.models.feature_extraction import get_graph_node_names
import random

class SignImageFolder(ImageFolder):
  def __init__(self, root, transform=None, resize=(224, 224), asl=False):
    self.transform = self._get_transform(transform, resize)
    super().__init__(root, transform=self.transform)
    self.idx_to_class = {v: k for k, v in self.class_to_idx.items()}
    self.dims = []
    self.asl = asl
    self.img_ids = [self._get_img_id(img_path, self.asl) for img_path, _ in self.samples]

  def _get_transform(self, transform, resize):
     return transform if transform else transforms.Compose([
        transforms.Resize(resize, transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                              std=[0.229, 0.224, 0.225]),
      ])

  def _get_img_id(self, image_path, asl):
    filename = os.path.basename(image_path)
    img_id=''
    if asl:
      img_id = filename.lower().replace('.jpg', '')
    else:
      try:
        img_id = int(filename.lower().split('_')[1].replace('.jpg', ''))
      except IndexError:
        print(f'Couldnt load {filename}')
    return img_id

  def __getitem__(self, index):
    img, label = super().__getitem__(index)
    img_path = self.samples[index][0]
    filename = os.path.basename(img_path)
    img_id = self._get_img_id(img_path, self.asl)

    return img, label, img_id

  def subset(self, sample_size=200, randomize=True):
    class_to_idx = self.class_to_idx
    idx_to_class = {v: k for k, v in class_to_idx.items()}
    class_indices = {class_name: [] for class_name in idx_to_class.values()}
    
    for idx, label in enumerate(self.targets):
        class_name = idx_to_class[label]
        class_indices[class_name].append(idx)
    
    sample = []
    for class_name, indices in class_indices.items():
        if randomize:
          random.shuffle(indices)
        sample.extend(indices[:sample_size])
    
    self.samples = [self.samples[i] for i in sample]
    self.targets = [self.targets[i] for i in sample]

  def denormalize(self, tensor):
    inv_normalize = transforms.Normalize(
        mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
        std=[1/0.229, 1/0.224, 1/0.225]
    )
    return inv_normalize(tensor)

  def plot(self, id):
      tensor, class_name, img_id = self.__getitem__(id)
      tensor = self.denormalize(tensor).clamp(0,1)
      image = tensor.permute(1,2,0).numpy()

      plt.imshow(image)
      plt.title(f"{self.idx_to_class[class_name]}:{img_id}")
      plt.show()


class ModelLoader:
  available_models = [
    'vgg19',
    'resnet50',
    'inception_v3',
    'efficientnet_b0',
    'efficientnet_b1',
    'efficientnet_b6'
  ]
  
  '''
    https://jacobgil.github.io/pytorch-gradcam-book/introduction.html

    Resnet18 and 50: model.layer4[-1]
    VGG and densenet161: model.features[-1]
    ViT: model.blocks[-1].norm1
    SwinT: model.layers[-1].blocks[-1].norm1

    Are suggested for last convolutional layers.

    Modify return_nodes_dict as needed. 
    more layers = more features = more space/memory
    Print self.nodes or self.model to see available layers
  '''
  return_nodes_dict = {
    'vgg19': ['features.34', 'features.35', 'features.36', 'avgpool', 'classifier.6'],
    'resnet50': ['layer4.2.conv3', 'layer4.2.relu_2', 'avgpool', 'fc'],
    'inception_v3': ['Mixed_7c.branch_pool.conv', 'avgpool', 'fc'],
    'efficientnet_b0': ['features.8', 'avgpool', 'classifier.1'],
    'efficientnet_b1': ['features.8', 'avgpool', 'classifier.1'],
    'efficientnet_b6': ['features.8', 'avgpool', 'classifier.1'],
  }

  fc_return_nodes = {
    'vgg19': ['classifier.6'],
    'resnet50': ['fc'],
    'inception_v3': ['fc'],
    'efficientnet_b0': ['classifier.1'],
    'efficientnet_b1': ['classifier.1'],
    'efficientnet_b6': ['classifier.1'],
  }

  def __init__(self, model_name, return_nodes_dict=None):
    assert model_name in self.available_models
    self.model_name = model_name
    self.model = self.load()
    self.return_nodes = return_nodes_dict[model_name] if return_nodes_dict else self.return_nodes_dict[model_name]
    self.device = device("cuda" if cuda.is_available() else "cpu")
    self.input_dim = self.input_dims[model_name]
    self.nodes = get_graph_node_names(self.model,
                                      suppress_diff_warning=True)[1] # eval nodes
  def load(self):
      if self.model_name == 'vgg19':
          from torchvision.models import vgg19, VGG19_Weights
          model = vgg19(weights=VGG19_Weights.DEFAULT)
      elif self.model_name == 'resnet50':
          from torchvision.models import resnet50, ResNet50_Weights
          model = resnet50(weights=ResNet50_Weights.DEFAULT)
      elif self.model_name == 'inception_v3':
          from torchvision.models import inception_v3, Inception_V3_Weights
          model = inception_v3(weights=Inception_V3_Weights.DEFAULT)
      elif self.model_name == 'efficientnet_b0':
          from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
          model = efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)
      elif self.model_name == 'efficientnet_b1':
          from torchvision.models import efficientnet_b1, EfficientNet_B1_Weights
          model = efficientnet_b1(weights=EfficientNet_B1_Weights.DEFAULT)
      elif self.model_name == 'efficientnet_b6':
          from torchvision.models import efficientnet_b6, EfficientNet_B6_Weights
          model = efficientnet_b6(weights=EfficientNet_B6_Weights.DEFAULT)
      model.eval()
      return model

  # https://discuss.pytorch.org/t/how-to-get-input-shape-of-model/85877/4
  input_dims = {
    'vgg19': (224, 224),
    'resnet50': (224, 224),
    'inception_v3': (299, 299),
    'efficientnet_b0': (224, 224),
    'efficientnet_b1': (240, 240),
    'efficientnet_b6': (528, 528),
  }