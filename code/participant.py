import tensorflow as tf
from tensorflow.keras.models import Model
import os
import numpy as np

class Participant:
    participants = {
        0: list(range(2914, 2951)),
        1: list(range(2871, 2904)),
        2: list(range(2323, 2356)),
        3: list(range(2285, 2314)),
        4: list(range(1646, 1675)),
        5: list(range(1503, 1535)) + list(range(1537, 1544)),
    }

    def __init__(self, num, loader, batch_size=32):
        self.num = num
        self.img_ids = self.participants[num]
        self.dataset = loader.dataset.filter(lambda img, label, img_id: tf.reduce_any(tf.equal(img_id, self.img_ids))).batch(batch_size)
        self.feats = {}
        self.labels = [loader.labels[loader.img_ids.index(img_id)] for img_id in self.img_ids]

    def _generate_filename(self, feat_path, model_name):
        return os.path.join(feat_path, f'p{self.num}_{model_name}_feats.npz')

    def encode_labels(self):
        label_map = {'A': 0, 'H': 1, 'L': 2, 'N': 3, 'O': 4, 'P': 5, 'R': 6}
        return np.array([label_map[label] for label in self.labels])
    
    def get_feats(self, model, layer_names, preprocess_input, feat_path=None):
        self.feats[model.name] = {layer_name: None for layer_name in layer_names}
        
        if feat_path:
          feat_filename = self._generate_filename(feat_path, model.name)
          if os.path.exists(feat_filename):
              print(f"Features already computed. Loading from {feat_filename}")
              loaded_feats = np.load(feat_filename)
              for layer_name in layer_names:
                  self.feats[model.name][layer_name] = loaded_feats[layer_name]
              return
        
        print(f'Getting feats for {model.name}')
        feature_model = Model(inputs=model.input, outputs=[model.get_layer(name).output for name in layer_names])

        for batch in self.dataset:
            img_batch, labels, ids = batch
            print(f'Doing forward pass for the batch with ids: {ids}')
            img_batch_resized = tf.image.resize(img_batch, model.input_shape[1:-1], method=tf.image.ResizeMethod.LANCZOS3)
            img_batch_preprocessed = preprocess_input(img_batch_resized)
            features = feature_model(img_batch_preprocessed, training=False)

            for layer_name, feature in zip(layer_names, features):
                feature_np = feature.numpy() 
                if self.feats[model.name][layer_name] is None:
                    self.feats[model.name][layer_name] = feature_np
                else:
                    self.feats[model.name][layer_name] = np.concatenate(
                        [self.feats[model.name][layer_name], feature_np], axis=0
                    )

        if feat_path:
            print(f"Saving features to {feat_filename}")
            np.savez(feat_filename, **{layer_name: self.feats[model.name][layer_name] for layer_name in layer_names})



def load_participants(loader, feat_path, nums=[0,1,2,3,4,5]):
  '''
    Forward passing to get all participant features. 
    If no feature file exists yet, get_feats() should 
    be called in separate cells for each participant/model 
    in order to not have any RAM issues
    (for efficientnetb6 batch_size should be minimum)
  '''
  p = []
  for n in nums:
    par = Participant(n, loader)
    for model_name in loader.models.keys():
      par.get_feats(*loader.models[model_name], feat_path)
    p.append(par)
  return p
