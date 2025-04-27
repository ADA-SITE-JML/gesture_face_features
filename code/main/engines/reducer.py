import numpy as np
import os
import matplotlib.pyplot as plt

import umap
from sklearn.manifold import TSNE
from sklearn.model_selection import ParameterGrid
from sklearn.decomposition import PCA

from config import MODEL_ZOO, FULL_PARAM_GRID

import warnings
warnings.simplefilter("ignore", FutureWarning)

class Reducer:
  def __init__(self, feats, targets, model_name, layer_name, method='umap', n_components=2, random_state=None, emb_path=None):
    self.feats = feats
    self.targets = targets
    self.model_name = model_name
    self.layer_name = layer_name
    self.random_state = random_state
    self.n_components = n_components
    self.embedding = None
    self.params = None
    self.method = method
    self.layer_feats = self._get_layer_feats()
    self.emb_path = emb_path

  def _get_layer_feats(self):
    layer_feats = self.feats[self.model_name][self.layer_name]
    return layer_feats.view(layer_feats.shape[0], -1)

  def _get_file_name(self):
    return  f"{self.method}_{self.model_name}_{self.layer_name}_rs_{str(self.random_state)}.npz"

  def _get_key_name(self, params):
    return ' '.join(str(value) for value in params.values())

  def get_embedding(self, params=None):
    # print(params)
    if self.method == 'umap':
      self.params = params or {
        'n_neighbors': 30,
        'min_dist': 0.3,
        'metric': 'euclidean',
      }
      reducer = umap.UMAP(
        n_components=self.n_components,
        n_neighbors=self.params['n_neighbors'],
        min_dist=self.params['min_dist'],
        metric=self.params['metric'],
        random_state=self.random_state
      )
    elif self.method == 'tsne':
      self.params = params or {
        'perplexity': 10,
        'learning_rate': 300,
        'metric': 'euclidean',
      }
      reducer = TSNE(
        n_components=self.n_components,
        perplexity=self.params['perplexity'],
        learning_rate=self.params['learning_rate'],
        metric=self.params['metric'],
        random_state=self.random_state
      )
    elif self.method == 'pca':
      self.params = params or {
          'whiten': False,
          'svd_solver': 'full',
      }
      reducer = PCA(
          n_components=self.n_components,
          whiten=self.params['whiten'],
          svd_solver=self.params['svd_solver'],
          random_state=self.random_state
      )

    self.embedding = reducer.fit_transform(self.layer_feats)

  def get_embedding_from_disk(self, params, verbose=False):
    if not self.emb_path or not params:
      print('No path or params to load form disk.')
      return

    file_name = self._get_file_name()
    key_name = self._get_key_name(params)
    file_path = os.path.join(self.emb_path, file_name)

    if os.path.exists(file_path):
      with np.load(file_path, allow_pickle=True) as f:
        data = dict(f)
        if key_name in data:
          self.embedding = data[key_name]
          self.params = params
          if verbose:
            print(f'Data loaded for {file_name}')
        else:
          self.get_embedding(params)
          data[key_name] = self.embedding
          np.savez(file_path, **data)
          if verbose:
            print(f"Data saved to {file_path}.")
        del data
    else:
      self.get_embedding(params)
      np.savez(file_path, **{key_name: self.embedding})
      if verbose:
        print(f"Data saved to {file_path}.")

  def plot(self, idx_to_class=None):
      scatter = plt.scatter(
        self.embedding[:, 0],
        self.embedding[:, 1],
        c=self.targets,
        cmap='viridis'
      )

      unique_targets = np.unique(self.targets)
      handles = []
      for target in unique_targets:
        letter = idx_to_class[target] if idx_to_class else target
        handle = plt.Line2D([0], [0], marker='o', color='w',
                            markerfacecolor=plt.cm.viridis(target / max(unique_targets)),
                            markersize=10, label=letter)
        handles.append(handle)

      plt.xticks([])
      plt.yticks([])
      plt.legend(handles=handles, title="Letters", loc='lower right')
      plt.show()

  def __repr__(self):
    return f"Embedding shape: {self.embedding.shape}, {self.method} params: {self.params})"

class ReducerEngine:
  def __init__(self, feats, targets, method='umap', param_grids=None, n_components=2, random_state=None, emb_path=None):
    self.feats = feats
    self.targets = np.array(targets)
    self.method = method
    self.param_grids = param_grids or FULL_PARAM_GRID
    self.n_components = n_components
    self.random_state = random_state
    self.reducers = []
    self.emb_path = emb_path

  def get_all_embeddings(self, from_disk=True):
    print(f'Getting {self.method} embeddings for grid {self.param_grids[self.method]}.')
    for model_name in MODEL_ZOO:
      for layer_name in MODEL_ZOO[model_name]['return_nodes']:
        if layer_name in self.feats[model_name].keys():
          # print(f'-------- {model_name} {layer_name} ---------')
          for params in ParameterGrid(self.param_grids[self.method]):
            reducer = Reducer(self.feats, self.targets, model_name, layer_name, self.method, self.n_components, self.random_state, self.emb_path)
            if from_disk:
              reducer.get_embedding_from_disk(params, verbose=False)
            else:
              reducer.get_embedding(params)
            self.reducers.append(reducer)