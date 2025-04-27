from PIL import Image
import matplotlib.pyplot as plt
import requests
from io import BytesIO
import json
import requests
import torch
from collections import Counter
import numpy as np
from config import MODEL_ZOO

def predict(model_name, feats):
  feat_tensor = get_fc_tensor(model_name, feats)
  preds = torch.nn.Softmax(dim=1)(feat_tensor)
  idx = torch.argmax(preds, dim=1)
  return preds, idx, feat_tensor

def get_fc_tensor(model_name, feats):
  # assumes that the fc is item
  # fc_layer = MODEL_ZOO[model_name]['return_nodes'][-1]
  fc_layer = list(feats[model_name].keys())[-1]
  # if fc_layer not in feats[model_name].keys():
  #   print(f'{fc_layer} not in feats')
  #   return None
  feat_tensor = feats[model_name][fc_layer]
  if feat_tensor.is_cuda:
    feat_tensor.cpu() 
  return feat_tensor

def decode_imagenet(pred_idx):
  url = "https://storage.googleapis.com/download.tensorflow.org/data/imagenet_class_index.json"
  class_idx = requests.get(url).json()
  predicted_labels = [class_idx[str(idx.item())][1] for idx in pred_idx]
  label_counts = dict(Counter(predicted_labels).most_common())
  return label_counts, predicted_labels


def delete_keys(data_dict, key_list):
  key_list = set(key_list)
  for k in list(data_dict.keys()):
    if k in key_list:
      del data_dict[k]