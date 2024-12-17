import torch
import os
from loader import ModelLoader

class FeatureExtractor:
    def __init__(self, modelloader, dataloader, feat_path):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.m = modelloader
        self.dataloader = dataloader
        self.feat_path = feat_path
        self.conv_layer_feats = {}
        self.gap_layer_feats = {}
        self.paths = {
          'conv': os.path.join(self.feat_path, f"{self.m.model_name}_conv_layer_feats.pt"),
          'gap':os.path.join(self.feat_path, f"{self.m.model_name}_gap_layer_feats.pt")
        }

    def _hook_fn_conv(self, module, input, output):
        for idx, img_id in enumerate(self.img_ids):
            self.conv_layer_feats[img_id.item()] = output[idx]

    def _hook_fn_gap(self, module, input, output):
        for idx, img_id in enumerate(self.img_ids):
            self.gap_layer_feats[img_id.item()] = output[idx]

    def extract_features(self, save):
        print(f'Extracting features for {self.m.model_name} layers.')
        self.m.model.to(self.device)

        self.hook1 = self.m.layers[0].register_forward_hook(self._hook_fn_conv)
        self.hook2 = self.m.layers[1].register_forward_hook(self._hook_fn_gap)
        
        for images, labels, img_ids in self.dataloader:
            self.img_ids = img_ids
            images = images.to(self.device)
            with torch.no_grad():
                output = self.m.model(images)
            print(f"Extracted ids for this batch: {img_ids}")
            
        self.hook1.remove()
        self.hook2.remove()

        if save:  
            self.save_features()

    def save_features(self):   
        torch.save(self.conv_layer_feats, self.paths['conv'])
        print(f"Conv layer features saved to {self.paths['conv']}")
        torch.save(self.gap_layer_feats, self.paths['gap'])
        print(f"Gap layer features saved to {self.paths['gap']}")

    def load_features(self, layer_type):        
        path = self.paths[layer_type]
        if os.path.exists(path):
            loaded_feats = torch.load(path, weights_only=True, map_location=self.device) 
            if layer_type == 'conv':
                self.conv_layer_feats = loaded_feats
            elif layer_type == 'gap':
                self.gap_layer_feats = loaded_feats
            print(f"Features loaded from {path}")
            return True
        else:
            print(f"No features found at {path}. For computing features call extract_features().")
            return False


def filter_ids(features, img_id_list):
  return {
      model_name: {
          img_id: feats
          for img_id, feats in img_dict.items()
          if img_id in img_id_list
      }
      for model_name, img_dict in features.items()
  }


def get_all_feats(dataloader, feat_path, model_name_list=ModelLoader.available_models, save=True):
    feats_conv = {}
    feats_gap = {}
    for model_name in model_name_list:
        m = ModelLoader(model_name)
        fe = FeatureExtractor(m, dataloader, feat_path)

        if not fe.load_features('conv'):
            fe.extract_features(save=save)
        feats_conv[model_name] = fe.conv_layer_feats
        if not fe.load_features('gap'):
            fe.extract_features(save=save)
        feats_gap[model_name] = fe.gap_layer_feats
    
    return feats_conv, feats_gap
