from PIL import Image
import matplotlib.pyplot as plt
import requests
from io import BytesIO
import json
import requests
import torch
from collections import Counter


def keep_fc(feats, MODEL_POOL, ModelLoader):
  for model_name in MODEL_POOL:
    delete_keys(feats[model_name], ModelLoader.fc_return_nodes[model_name])


def delete_keys(data_dict, keep_key_list):
  '''To free RAM when having nested dictionary'''
  key_list = set(keep_key_list)
  for k in list(data_dict.keys()):
    if k not in key_list:
      del data_dict[k]


def decode_imagenet(pred_idx):
  url = "https://storage.googleapis.com/download.tensorflow.org/data/imagenet_class_index.json"
  class_idx = requests.get(url).json()
  predicted_labels = [class_idx[str(idx.item())][1] for idx in pred_idx]
  label_counts = dict(Counter(predicted_labels).most_common())
  return label_counts, predicted_labels


def get_predictions(fc):
  preds = torch.nn.Softmax(dim=1)(fc)
  idx = torch.argmax(preds, dim=1)
  return preds, idx

def fetch_img(url):
    try:
        response = requests.get(url)
        if response.status_code == 200:
            img = Image.open(BytesIO(response.content))
            return img
        else:
            print(f"Failed to retrieve image. Status code: {response.status_code}")
            return None
    except Exception as e:
        print(f"An error occurred: {e}")
        return None



