import itertools
from collections import Counter
import numpy as np
from typing import Optional, Dict, List
from engines.embedding import Embedding, EmbeddingEngine

class Cluster:
  def __init__(self, embedding: Embedding, idx_to_class: Optional[Dict[int, str]] = None):
    """
    Scores all clusters based on custom evaluation.

    Args:
      embedding: Embedding object with attributes 'cluster' and 'reducer'.
      idx_to_class: Optional dictionary mapping indices to class labels.
    """
    self.embedding = embedding
    self.idx_to_class = idx_to_class or {
        0: 'A', 1: 'H', 2: 'L',
        3: 'N', 4: 'O', 5: 'P', 6: 'R'
    }
    self.similarity_scores = self._build_similarity_scores()
    self.cluster_groups = self.group()
    self.score = self.evaluate()
    self.purity = self.get_purity()
    self.nleep_score = 0

  def _build_similarity_scores(self) -> Dict[tuple, float]:
    """
    Builds a similarity score dictionary for letter pairs.

    Returns:
      Dictionary of pairwise similarity scores.
    """
    similarity_scores = {
      ("R", "N"): 0.8,
      ("H", "P"): 0.8,
      ("P", "L"): 0.8,
      ("R", "O"): 0.6,
      ("N", "O"): 0.6,
      ("H", "L"): 0.6,
    }
    similarity_scores.update({(b, a): score for (a, b), score in similarity_scores.items()})
    for letter in 'AHLNPRO':
        similarity_scores[(letter, letter)] = 1.0
    return similarity_scores

  def evaluate(self) -> float:
    """
    Evaluates overall cluster quality based on similarity scores.

    Returns:
      Average similarity score across all clusters.
    """
    count = self.embedding.n_clusters
    total = sum(self.eval_single(i) for i in range(count))
    return total / count

  def eval_single(self, cluster_num: int) -> float:
    """
    Evaluates a single cluster's quality.

    Args:
      cluster_num: Cluster number to evaluate.

    Returns:
      Average similarity score within the cluster.
    """
    letters = list(self.cluster_groups.values())[cluster_num]
    if len(letters) == 1:
      return 0
    pairs = list(itertools.combinations(letters, 2))
    score = sum(self.similarity_scores.get(pair, 0) for pair in pairs) / len(pairs)
    return score

  def group(self) -> Dict[int, List[str]]:
    """
    Groups samples by their cluster labels.

    Returns:
      Dictionary mapping cluster numbers to lists of letter labels.
    """
    data = np.column_stack((self.embedding.cluster, self.embedding.reducer.targets))
    cluster_groups = {}
    for cluster, target in data:
      cluster_groups.setdefault(cluster, []).append(self.idx_to_class[target])
    return cluster_groups

  def get_purity(self) -> float:
    """
    Returns:
      Purity score between 0 and 1.
    """
    total_samples = len(self.embedding.reducer.targets)
    return sum(max(Counter(letter_group).values()) for letter_group in self.cluster_groups.values()) / total_samples

  def __repr__(self) -> str:
    cluster_details = f"-------{self.embedding.reducer.model_name} "
    cluster_details += f"{self.embedding.reducer.layer_name}-------\n\n"
    for cluster, letter_group in self.cluster_groups.items():
      cluster_details += f"Cluster {cluster}: {letter_group}\n"
    cluster_details += f"\nPurity: {self.purity:.2f}\n"
    cluster_details += f"Custom Weighted Score: {self.score:.2f}\n"
    return cluster_details


class ClusterEngine:
  def __init__(self, embedding_engine: EmbeddingEngine):
    """
    Engine to score and find the best cluster from a collection of embeddings.

    Args:
      embedding_engine: An object containing a list of clustered embeddings.
    """
    self.embedding_engine = embedding_engine
    self.scored_clusters: List[Cluster] = []
    self.best_cluster: Optional[Cluster] = None
    self.best_score: float = 0

  def score_all_clusters(self) -> None:
    '''
    Scores all clusters and selects the best one based on custom evaluation.
    '''

    print(f'Getting cluster scores...')
    for embedding in self.embedding_engine.clusters:
      cluster = Cluster(embedding)
      self.scored_clusters.append(cluster)
      if cluster.score > self.best_score:
        self.best_score = cluster.score
        self.best_cluster = cluster
