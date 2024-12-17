import matplotlib.pyplot as plt
from torchvision import transforms
from torchvision.datasets import ImageFolder
import os
from torch import device, cuda
from torchvision.models.feature_extraction import get_graph_node_names

class SignImageFolder(ImageFolder):
  def __init__(self, root, transform=transforms.Compose([
        transforms.Resize((600, 600), transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                              std=[0.229, 0.224, 0.225]),
      ])):
    super().__init__(root, transform=transform)
    self.idx_to_class = {v: k for k, v in self.class_to_idx.items()}
    self.dims = []
    self.img_ids = [self._get_img_id(img_path) for img_path, _ in self.samples]
    
  def _get_img_id(self, image_path):
      filename = os.path.basename(image_path)
      return int(filename.split('_')[1].replace('.JPG', ''))

  def __getitem__(self, index):
    img, label = super().__getitem__(index)
    img_path = self.samples[index][0]
    filename = os.path.basename(img_path)
    img_id = int(filename.split('_')[1].replace('.JPG', ''))

    return img, label, img_id

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

  def __init__(self, model_name):
    assert model_name in self.available_models
    self.model_name = model_name
    self.model = self.load()
    self.return_nodes = self.return_nodes_dict[model_name]
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