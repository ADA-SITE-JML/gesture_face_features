import os
import torch
from torchvision.models.feature_extraction import create_feature_extractor
from loader import ModelLoader
import numpy as np
from utils import get_predictions


def get_f_y_pred(model_name, feats, targets):
  '''
    assumes that the fc is last key
  '''
  fc_layer = list(feats[model_name].keys())[-1]
  feat_tensor = feats[model_name][fc_layer]
  feat_tensor = feat_tensor.cpu() if feat_tensor.is_cuda else feat_tensor
  f = feat_tensor.numpy()
  y = np.array(targets)
  preds, idx = get_predictions(feat_tensor)
  return f, y, preds, idx
  

def filter_by_id(feats, dataset, img_ids):
    indices = [dataset.img_ids.index(img_id) for img_id in img_ids]
    
    filtered_feats = {}
    for model_name, layers in feats.items():
        filtered_feats[model_name] = {}
        
        for layer_name, feat in layers.items():
            filtered_feats[model_name][layer_name] = feat[indices]
    
    return filtered_feats

def extract_model_features(dataloader, model_name_list=ModelLoader.available_models, feat_path=None, return_nodes=None):
  feats = {}
  for model_name in model_name_list:
    feats[model_name] = extract_features(dataloader, ModelLoader(model_name, return_nodes), feat_path)
  return feats

def extract_features(dataloader, modelloader, feat_path=None, batch=False):
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

  fe = create_feature_extractor(model, return_nodes=return_nodes, 
                                suppress_diff_warning=True).to(device)

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