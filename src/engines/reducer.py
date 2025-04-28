import numpy as np
import os
import matplotlib.pyplot as plt
import umap
from sklearn.manifold import TSNE
from sklearn.model_selection import ParameterGrid
from sklearn.decomposition import PCA
from typing import Optional, Dict, Any, List, Union
from config import MODEL_ZOO, FULL_PARAM_GRID
import itertools

import warnings
warnings.simplefilter("ignore", FutureWarning)

class Reducer:
  def __init__(self, feats: Dict[str, Dict[str, Any]], targets: np.ndarray, model_name: str, layer_name: str,
                method: str = 'umap', n_components: int = 2, random_state: Optional[int] = None,
                emb_path: Optional[str] = None):
    """
    Initializes a dimensionality reduction object for a specific model layer.

    Args:
      feats: Nested dictionary of extracted features.
      targets: Array of class labels.
      model_name: Model name key.
      layer_name: Layer name key.
      method: Dimensionality reduction method ('pca', 'umap', 'tsne').
      n_components: Number of dimensions to reduce to.
      random_state: Random seed for reproducibility.
      emb_path: Optional path to save/load embeddings.
    """
    self.feats = feats
    self.targets = targets
    self.model_name = model_name
    self.layer_name = layer_name
    self.random_state = random_state
    self.n_components = n_components
    self.embedding: Optional[np.ndarray] = None
    self.params: Optional[Dict[str, Any]] = None
    self.method = method
    self.layer_feats = self._get_layer_feats()
    self.emb_path = emb_path

  def _get_layer_feats(self) -> np.ndarray:
    """
    Retrieves and flattens features for the given model layer.
    """
    layer_feats = self.feats[self.model_name][self.layer_name]
    return layer_feats.view(layer_feats.shape[0], -1)

  def _get_file_name(self) -> str:
    """
    Generates a filename based on model, layer, method, and random state.
    """
    return f"{self.method}_{self.model_name}_{self.layer_name}_rs_{str(self.random_state)}.npz"

  def _get_key_name(self, params: Dict[str, Any]) -> str:
    """
    Generates a key name based on parameter dictionary.
    """
    return ' '.join(str(value) for value in params.values())

  def get_embedding(self, params: Optional[Dict[str, Any]] = None) -> None:
    """
    Computes the embedding from the feature matrix based on the specified method.

    Args:
      params: Optional parameters for the reduction method.
    """
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

  def get_embedding_from_disk(self, params: Dict[str, Any], verbose: bool = False) -> None:
      """
      Loads an embedding from disk if available, otherwise computes and saves it (if path provided).

      Args:
        params: Parameters used to find the saved embedding.
        verbose: Whether to print messages about loading/saving.
      """
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

  def plot(self, idx_to_class: Optional[Dict[int, str]] = None) -> None:
    """
    Plots the embedding points, separating classes by color and shape.

    Args:
      idx_to_class: Optional mapping from class index to label name.
    """

    markers = itertools.cycle(('o', 's', '^', 'v', '>', '<', 'p', '*', 'h', 'D'))
    unique_targets = np.unique(self.targets)
    color_map = plt.cm.get_cmap('viridis', len(unique_targets))

    plt.figure(figsize=(8, 6))
    handles = []

    for i, target in enumerate(unique_targets):
      marker = next(markers)
      color = color_map(i)

      idx = self.targets == target
      plt.scatter(
        self.embedding[idx, 0],
        self.embedding[idx, 1],
        color=color,
        marker=marker,
        label=idx_to_class[target] if idx_to_class else str(target),
        s=50
      )

      handle = plt.Line2D(
        [0], [0],
        marker=marker,
        color='w',
        markerfacecolor=color,
        markeredgecolor='k',
        markersize=10,
        label=idx_to_class[target] if idx_to_class else str(target)
      )
      handles.append(handle)

    plt.xticks([])
    plt.yticks([])
    plt.legend(
      handles=handles,
      title="Letters",
      bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.
    )
    plt.tight_layout()
    plt.show()

  def __repr__(self) -> str:
    return f"Embedding shape: {self.embedding.shape}, {self.method} params: {self.params})"


class ReducerEngine:
  def __init__(self, feats: Dict[str, Dict[str, Any]], targets: Union[np.ndarray, List[int]],
              method: str = 'umap', param_grids: Optional[Dict[str, Any]] = None,
              n_components: int = 2, random_state: Optional[int] = None, emb_path: Optional[str] = None):
    """
    Manages multiple Reducer instances for different layers and parameters.

    Args:
      feats: Nested dictionary of extracted features.
      targets: Class label array.
      method: Dimensionality reduction method.
      param_grids: Grid of parameters for each method.
                   Will default to FULL_PARAM_GRID from config.py
      n_components: Number of dimensions to reduce to.
      random_state: Random seed for reproducibility.
      emb_path: Optional path to save/load embeddings.
    """
    self.feats = feats
    self.targets = np.array(targets)
    self.method = method
    self.param_grids = param_grids or FULL_PARAM_GRID
    self.n_components = n_components
    self.random_state = random_state
    self.reducers: List[Reducer] = []
    self.emb_path = emb_path

  def get_all_embeddings(self, from_disk: bool = True) -> None:
    """
    Generates embeddings for all models and layers across the parameter grid.

    Args:
      from_disk: Whether to load from disk if available.
    """
    
    print(f'Getting {self.method} embeddings for grid {self.param_grids[self.method]}.')
    for model_name in MODEL_ZOO:
      for layer_name in MODEL_ZOO[model_name]['return_nodes']:
        if layer_name in self.feats[model_name].keys():
          for params in ParameterGrid(self.param_grids[self.method]):
            reducer = Reducer(
              self.feats, self.targets, 
              model_name, layer_name, 
              self.method, self.n_components, 
              self.random_state, self.emb_path
            )
            if from_disk:
              reducer.get_embedding_from_disk(params, verbose=False)
            else:
              reducer.get_embedding(params)
            self.reducers.append(reducer)