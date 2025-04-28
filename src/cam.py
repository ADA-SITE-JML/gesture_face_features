import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
from typing import List, Tuple

def register_hooks(model: torch.nn.Module, target_layer: str) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[torch.utils.hooks.RemovableHandle]]:
  """
  Registers forward and backward hooks on a given layer to capture feature maps and gradients.

  Args:
    model: The model containing the target layer.
    target_layer: Name of the layer to hook.

  Returns:
    feature_maps: List of forward feature maps.
    gradients: List of backward gradients.
    handles: List of hook handles for later removal.
  """
  feature_maps, gradients = [], []
  layer = dict(model.named_modules())[target_layer]
  handles = [
    layer.register_forward_hook(lambda m, i, o: feature_maps.append(o)),
    layer.register_backward_hook(lambda m, gi, go: gradients.append(go[0]))
  ]
  return feature_maps, gradients, handles

def compute_gradcam(feature_maps: List[torch.Tensor], gradients: List[torch.Tensor]) -> np.ndarray:
  """
  Computes a normalized Grad-CAM heatmap from feature maps and gradients.

  Args:
    feature_maps: List containing forward feature maps.
    gradients: List containing backward gradients.

  Returns:
    Grad-CAM heatmap as a NumPy array.
  """
  feature, grad = feature_maps[0], gradients[0]
  feature = feature.squeeze(0) if feature.dim() == 4 else feature
  grad = grad.squeeze(0) if grad.dim() == 4 else grad
  weights = grad if grad.dim() == 1 else grad.mean(dim=1 if grad.dim() == 2 else (1,2))
  cam = torch.sum(weights.view(-1, 1, 1) * feature, dim=0)
  cam = torch.relu(cam)
  cam = (cam - cam.min()) / (cam.max() - cam.min())
  return cam.detach().cpu().numpy()

def superimpose_heatmap(img: np.ndarray, cam: np.ndarray, alpha: float = 0.3) -> np.ndarray:
  """
  Superimposes a Grad-CAM heatmap onto an image.

  Args:
    img: Original image in BGR format.
    cam: Grad-CAM heatmap.
    alpha: Transparency factor for heatmap overlay.

  Returns:
    Superimposed BGR image as a NumPy array.
  """
  cam_resized = cv2.resize(cam, (img.shape[1], img.shape[0]))
  heatmap = cv2.applyColorMap(np.uint8(255 * cam_resized), cv2.COLORMAP_JET)
  return cv2.addWeighted(img, 1 - alpha, heatmap, alpha, 0)

def generate_gradcam(model: torch.nn.Module, img_input: torch.Tensor, target_layer: str) -> np.ndarray:
  """
  Generates a Grad-CAM heatmap for a given input and model.

  Args:
    model: The model to evaluate.
    img_input: Input tensor (batched).
    target_layer: Name of the target convolutional layer.

  Returns:
    Grad-CAM heatmap as a NumPy array.
  """
  feature_maps, gradients, handles = register_hooks(model, target_layer)
  output = model(img_input)
  pred_class = output.argmax(dim=1)
  model.zero_grad()
  output[0, pred_class].backward()
  cam = compute_gradcam(feature_maps, gradients)
  for h in handles:
    h.remove()
  return cam

def plot_gradcam(cam: np.ndarray, superimposed_img: np.ndarray) -> None:
  """
  Plots the Grad-CAM heatmap and the superimposed image side by side.

  Args:
    cam: Grad-CAM heatmap.
    superimposed_img: BGR image with heatmap overlay.
  """
  plt.figure(figsize=(10,5))
  plt.subplot(1,2,1)
  plt.imshow(cam, cmap='jet')
  plt.axis('off')
  plt.subplot(1,2,2)
  plt.imshow(cv2.cvtColor(superimposed_img, cv2.COLOR_BGR2RGB))
  plt.axis('off')
  plt.tight_layout()
  plt.show()