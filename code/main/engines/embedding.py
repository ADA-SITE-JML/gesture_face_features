import matplotlib.pyplot as plt
import numpy as np

from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture

class Embedding:
  def __init__(self, reducer, model='kmeans', n_clusters=None, random_state=None):
    self.reducer = reducer
    self.random_state = random_state
    self.model = model
    self.cluster = None
    self.n_clusters = n_clusters or len(np.unique(self.reducer.targets))

  def get_clusters(self):
    if self.model == 'kmeans':
      model = KMeans(n_clusters=self.n_clusters, random_state=self.random_state)
    elif self.model == 'gmm':
      model = GaussianMixture(n_components=self.n_clusters, random_state=self.random_state)
    self.cluster = model.fit_predict(self.reducer.embedding)

  def __repr__(self):
    return f"Embedding for {self.model}"

class EmbeddingEngine:
  def __init__(self, reducer_engine, model='kmeans', n_clusters=None, random_state=None):
    self.reducer_engine = reducer_engine
    self.random_state = random_state
    self.model = model
    self.clusters = []
    self.n_clusters = n_clusters

  def get_all_clusters(self):
    print(f'Getting {self.model} clusters...')
    for reducer in self.reducer_engine.reducers:
      # print(f'-------- {reducer.model_name} {reducer.layer_name} ---------')
      cluster = Embedding(reducer, self.model, self.n_clusters, self.random_state)
      cluster.get_clusters()
      self.clusters.append(cluster)