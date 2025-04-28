import requests
import torch
from collections import Counter
from typing import Tuple, Dict, List, Any
from config import MODEL_ZOO

def predict(model_name: str, feats: Dict[str, Dict[str, torch.Tensor]]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Predicts the class probabilities and indices for the given model features.

    Args:
      model_name: Name of the model.
      feats: Dictionary containing model features.

    Returns:
      Tuple of (predictions, predicted indices, feature tensor).
    """
    feat_tensor = get_fc_tensor(model_name, feats)
    preds = torch.nn.Softmax(dim=1)(feat_tensor)
    idx = torch.argmax(preds, dim=1)
    return preds, idx, feat_tensor

def get_fc_tensor(model_name: str, feats: Dict[str, Dict[str, torch.Tensor]]) -> torch.Tensor:
    """
    Retrieves the fully connected (FC) layer tensor for the given model.

    Args:
      model_name: Name of the model.
      feats: Dictionary containing model features.

    Returns:
      The feature tensor corresponding to the last layer.
    """
    fc_layer = list(feats[model_name].keys())[-1]
    feat_tensor = feats[model_name][fc_layer]
    if feat_tensor.is_cuda:
        feat_tensor = feat_tensor.cpu()
    return feat_tensor

def decode_imagenet(pred_idx: torch.Tensor) -> Tuple[Dict[str, int], List[str]]:
    """
    Decodes ImageNet prediction indices into human-readable labels.

    Args:
      pred_idx: Tensor of predicted class indices.

    Returns:
      A tuple containing:
        - A dictionary with label counts.
        - A list of predicted labels.
    """
    url = "https://storage.googleapis.com/download.tensorflow.org/data/imagenet_class_index.json"
    class_idx = requests.get(url).json()
    predicted_labels = [class_idx[str(idx.item())][1] for idx in pred_idx]
    label_counts = dict(Counter(predicted_labels).most_common())
    return label_counts, predicted_labels

def delete_keys(data_dict: Dict[Any, Any], key_list: List[Any]) -> None:
    """
    Deletes specified keys from a dictionary.
    """
    key_list = set(key_list)
    for k in list(data_dict.keys()):
      if k in key_list:
        del data_dict[k]