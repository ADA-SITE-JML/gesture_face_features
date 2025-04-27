import os
import torch
import numpy as np
from typing import Optional, Dict, Union

import torch
from torchvision.models.feature_extraction import create_feature_extractor
from torch.utils.data import DataLoader
from model_loader import ModelLoader

def extract_features(
    dataloader: DataLoader,
    modelloader: ModelLoader,
    feat_path: Optional[str] = None,
    batch: bool = False
) -> Dict[str, Union[torch.Tensor, list[torch.Tensor]]]:
    """
    Extracts features from a model layer return nodes (defined in config.py) for a given dataloader.

    Args:
      dataloader (torch.utils.data.DataLoader): DataLoader providing (images, labels, img_ids).
      modelloader (ModelLoader): An object containing the model, model name, return nodes, and device.
      feat_path (str, optional): Directory to save or load precomputed features. If given, attempts to load features first.
      batch (bool, optional): If False, concatenates the extracted feature tensors. If True, keeps features in batch form.

    Returns:
      dict: A dictionary mapping each return node to its corresponding feature tensors.
    """
    model = modelloader.model
    model_name = modelloader.model_name
    return_nodes = modelloader.return_nodes
    device = modelloader.device

    if feat_path:
      path = os.path.join(feat_path, f'{model_name}_feats.pt')
      if os.path.exists(path):
        feats = torch.load(path, weights_only=True, map_location=device)
        print(f"Features loaded from {path}.")
        return feats

    fe = create_feature_extractor(model, return_nodes=return_nodes, suppress_diff_warning=True).to(device)

    feats = {layer_name: [] for layer_name in return_nodes}

    print(f'Extracting features for {model_name} {return_nodes}')
    for images, labels, img_ids in dataloader:
      images = images.to(device)
      with torch.no_grad():
        out = fe(images)
      print(f'Extracted features for ids {img_ids}')

      for layer_name, feature in out.items():
        feats[layer_name].append(feature.cpu())

    if not batch:
      for layer_name, feature_list in feats.items():
        feats[layer_name] = torch.cat(feature_list, dim=0)

    if feat_path:
      torch.save(feats, path)
      print(f'Saved features to {path}')

    return feats