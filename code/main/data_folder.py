import matplotlib.pyplot as plt
from torchvision import transforms
from torchvision.datasets import ImageFolder
import os
import torch
import numpy as np
from torch import Tensor
from typing import Optional, Tuple, List

class SignImageFolder(ImageFolder):
  """
  Custom dataset for sign language images with preprocessing, ID extraction, 
  and utility functions for normalization and visualization.

  Provided mean and std are ImageNet normalization values 
  required for pre-trained models.

  Overrides __getitem__ to return img_id as well. 
  Has plotting capacity.
  """
  def __init__(
    self,
    root: str,
    transform: Optional[transforms.Compose] = None,
    resize: Tuple[int, int] = (224, 224),
    mean: List[float] = [0.485, 0.456, 0.406],
    std: List[float] = [0.229, 0.224, 0.225]
  ):
    self.mean = torch.tensor(mean).view(3, 1, 1)
    self.std = torch.tensor(std).view(3, 1, 1)
    self.transform = self._get_transform(transform, resize)
    super().__init__(root, transform=self.transform)
    self.idx_to_class = {v: k for k, v in self.class_to_idx.items()}
    self.img_ids = [self._get_img_id(img_path) for img_path, _ in self.samples]

  def _get_transform(
    self,
    transform: Optional[transforms.Compose],
    resize: Tuple[int, int]
  ) -> transforms.Compose:
    if transform:
      return transform
    return transforms.Compose([
      transforms.Resize(resize, transforms.InterpolationMode.BICUBIC),
      transforms.ToTensor(),
      transforms.Normalize(mean=self.mean.flatten().tolist(), std=self.std.flatten().tolist())
    ])

  def _get_img_id(self, image_path: str) -> int:
    try:
      filename = os.path.basename(image_path)
      img_id = int(filename.lower().split('_')[1].replace('.jpg', ''))
      return img_id
    except (IndexError, ValueError):
      return -1

  def __getitem__(self, index: int) -> Tuple[Tensor, int, int]:
    img, label = super().__getitem__(index)
    img_path = self.samples[index][0]
    img_id = self._get_img_id(img_path)
    return img, label, img_id

  def denormalize(self, img: Tensor) -> Tensor:
    return img * self.std + self.mean

  def normalize(self, img: Tensor) -> Tensor:
    return (img - self.mean) / self.std

  def tensor_to_numpy(self, tensor: Tensor) -> np.ndarray:
    tensor = self.denormalize(tensor).clamp(0, 1)
    image = tensor.permute(1, 2, 0).numpy()
    return image

  def plot(self, id: int) -> None:
    tensor, class_name, img_id = self.__getitem__(id)
    image = self.tensor_to_numpy(tensor)
    plt.imshow(image)
    plt.title(f"{self.idx_to_class[class_name]}:{img_id}")
    plt.show()