import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture

# from LogME.LEEP import LEEP

from engines.embedding import EmbeddingEngine
from engines.cluster import ClusterEngine

from tllib.ranking.leep import log_expected_empirical_prediction

class Engine:
  def __init__(self, reducer_engine, model='kmeans', n_clusters=None, random_state=None):
    self.re = reducer_engine
    self.ee = None
    self.ce = None
    self.model = model
    self.random_state = random_state
    self.ranked_clusters = {}
    self.n_clusters = n_clusters

  def run(self, model_name_list):
    self.score(verbose=False)
    self.rank(model_name_list, 'score')
    self.rank(model_name_list, 'nleep')

  def score(self, verbose=True):
    self.ee = EmbeddingEngine(self.re, self.model, self.n_clusters, self.random_state)
    self.ee.get_all_clusters()
    self.ce = ClusterEngine(self.ee)
    self.ce.score_all_clusters()
    self.get_nleep_scores()
    if verbose:
      print(self.ce.best_cluster)
      self.ce.best_cluster.embedding.reducer.plot()

  def get_nleep_scores(self, factor=1):
    print('Getting cluster nleep scores...')
    for cluster in self.ce.scored_clusters:
      self.get_nleep_score(cluster, factor)

  def get_nleep_score(self, cluster=None, factor=1, random_state=None):
    cluster = cluster or self.ce.best_cluster
    n_components = cluster.embedding.n_clusters * factor
    embedding = cluster.embedding.reducer.embedding
    random_state = random_state or self.random_state
    targets = cluster.embedding.reducer.targets

    gmm = GaussianMixture(n_components=n_components, random_state=random_state)
    gmm.fit(embedding)
    gmm_predictions = gmm.predict_proba(embedding)
    # cluster.nleep_score = LEEP(gmm_predictions, targets)
    cluster.nleep_score = log_expected_empirical_prediction(gmm_predictions, targets)
    return cluster.nleep_score

  def corr(self):
    self.nleep_scores = [cluster.nleep_score for cluster in self.ce.scored_clusters]
    self.scores = [cluster.score for cluster in self.ce.scored_clusters]
    correlation = np.corrcoef(self.scores, self.nleep_scores)[0, 1]
    print(f"Pearson Correlation: {correlation}")
    plt.scatter(self.scores, self.nleep_scores)

  def get_best_cluster(self, model_name, method='score'):
    best_score = float('-inf')
    best_cluster = None
    for cluster in self.ce.scored_clusters:
      if model_name == cluster.embedding.reducer.model_name:
        if method == 'score':
          score = cluster.score
        elif method == 'nleep':
          score = cluster.nleep_score
        else:
          raise ValueError('Method must be "nleep" or "score".')
        if score > best_score:
          best_score = score
          best_cluster = cluster
    return best_cluster

  def rank(self, model_name_list, method='score'):
    clusters = []
    for model_name in model_name_list:
      best_cluster = self.get_best_cluster(model_name, method)
      clusters.append(best_cluster)
    clusters = sorted(clusters, key=lambda cluster: cluster.nleep_score if method == 'nleep' else cluster.score, reverse=True)
    self.ranked_clusters[method] = clusters

  def plot_ranking(self, method='nleep'):
    rows = []
    for cluster in self.ranked_clusters[method]:
      score = cluster.nleep_score if method == 'nleep' else cluster.score
      model_name = cluster.embedding.reducer.model_name
      layer_name = cluster.embedding.reducer.layer_name
      params = cluster.embedding.reducer.params
      mean = self.get_model_mean(model_name, method)
      rows.append({'model': model_name, 'layer': layer_name, 'params': params,
                  method: round(score, 2), 'mean': mean})
    return pd.DataFrame(rows)

  def get_model_mean(self, model_name, method='nleep'):
    sum = 0
    count = 0
    for cluster in self.ce.scored_clusters:
      if model_name == cluster.embedding.reducer.model_name:
        score = cluster.nleep_score if method == 'nleep' else cluster.score
        sum += score
        count += 1
    return sum / count
