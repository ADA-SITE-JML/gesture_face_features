from typing import Dict, Any, List
from torch import device as torch_device, cuda
from torchvision.models.feature_extraction import get_graph_node_names
import torch.nn as nn
from config import MODEL_ZOO

class ModelLoader:
  """
  A loader class for pretrained models from torchvision, handling device assignment, 
  model loading, and graph node extraction for feature extraction purposes.
  
  Attributes:
    model_name (str): Name of the model to load.
    model (nn.Module): The loaded pretrained model.
    return_nodes (Dict[str, str]): Mapping of nodes for feature extraction.
    device (torch.device): Device where the model is loaded ('cuda' or 'cpu').
    input_dim (int): Expected input dimension for the model.
    nodes (List[str]): List of node names available for feature extraction.
  """
  def __init__(self, model_name: str) -> None:
    assert model_name in MODEL_ZOO
    self.model_name: str = model_name
    self.model: nn.Module = self.load()
    self.return_nodes: Dict[str, str] = MODEL_ZOO[model_name]['return_nodes']
    self.device: torch_device = torch_device("cuda" if cuda.is_available() else "cpu")
    self.input_dim: int = MODEL_ZOO[model_name]['input_dim']
    self.nodes: List[str] = get_graph_node_names(self.model, suppress_diff_warning=True)[1]

  def load(self) -> nn.Module:
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
    else:
      raise ValueError(f"Model '{self.model_name}' is not supported.")
    model.eval()
    return model
