import matplotlib.pyplot as plt
import numpy as np
from typing import Optional, List

from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture

class Embedding:
  def __init__(self, reducer: any, model: str = 'kmeans', n_clusters: Optional[int] = None, random_state: Optional[int] = None):
    """
    Initializes an Embedding object for clustering reduced data.

    Args:
      reducer: A reducer object containing embeddings and targets.
      model: Clustering model to use ('kmeans' or 'gmm').
      n_clusters: Number of clusters. If None, inferred from unique targets.
      random_state: Random seed for reproducibility.
    """
    self.reducer = reducer
    self.random_state = random_state
    self.model = model
    self.cluster = None
    self.n_clusters = n_clusters or len(np.unique(self.reducer.targets))

  def get_clusters(self) -> None:
    """
    Performs clustering on the reducer's embedding and stores cluster labels.
    """
    if self.model == 'kmeans':
      model = KMeans(n_clusters=self.n_clusters, random_state=self.random_state)
    elif self.model == 'gmm':
      model = GaussianMixture(n_components=self.n_clusters, random_state=self.random_state)
    self.cluster = model.fit_predict(self.reducer.embedding)

  def __repr__(self) -> str:
    return f"Embedding for {self.model}"

class EmbeddingEngine:
  def __init__(self, reducer_engine: any, model: str = 'kmeans', n_clusters: Optional[int] = None, random_state: Optional[int] = None):
    """
    Initializes an EmbeddingEngine to manage clustering across multiple reducers.

    Args:
      reducer_engine: An object containing a list of reducers.
      model: Clustering model to use ('kmeans' or 'gmm').
      n_clusters: Number of clusters to use. If None, use default.
      random_state: Random seed for reproducibility.
    """
    self.reducer_engine = reducer_engine
    self.random_state = random_state
    self.model = model
    self.clusters: List[Embedding] = []
    self.n_clusters = n_clusters

  def get_all_clusters(self) -> None:
    """
    Applies clustering across all reducers in the reducer engine.
    """
    print(f'Getting {self.model} clusters...')
    for reducer in self.reducer_engine.reducers:
      cluster = Embedding(reducer, self.model, self.n_clusters, self.random_state)
      cluster.get_clusters()
      self.clusters.append(cluster)