import pandas as pd
import numpy as np
from typing import List, Optional, Any
from config import MODEL_ZOO

def get_cluster_score(cluster: Any, metric: str) -> float:
    return cluster.nleep_score if metric == 'nleep' else cluster.score

def get_best_cluster(scored_clusters: List[Any], model_name: str, metric: str = 'score') -> Optional[Any]:
    clusters = [c for c in scored_clusters if c.embedding.reducer.model_name == model_name]
    if not clusters:
        return None
    return max(clusters, key=lambda c: get_cluster_score(c, metric))

def rank_clusters(scored_clusters: List[Any], metric: str = 'nleep') -> List[Any]:
    ranked = [
        get_best_cluster(scored_clusters, model_name, metric)
        for model_name in MODEL_ZOO
    ]
    ranked = [c for c in ranked if c is not None]
    return sorted(ranked, key=lambda c: get_cluster_score(c, metric), reverse=True)

def get_model_mean(scored_clusters: List[Any], model_name: str, metric: str = 'nleep') -> float:
    scores = [
        get_cluster_score(c, metric)
        for c in scored_clusters
        if c.embedding.reducer.model_name == model_name
    ]
    return np.mean(scores) if scores else 0.0

def plot_ranking(scored_clusters: List[Any], ranked_clusters: List[Any], metric: str = 'nleep') -> pd.DataFrame:
    rows = []
    for cluster in ranked_clusters:
        model_name = cluster.embedding.reducer.model_name
        layer_name = cluster.embedding.reducer.layer_name
        params = cluster.embedding.reducer.params
        score_value = get_cluster_score(cluster, metric)
        mean_value = get_model_mean(scored_clusters, model_name, metric)
        rows.append({
            'model': model_name,
            'layer': layer_name,
            'params': params,
            metric: round(score_value, 2),
            'mean': round(mean_value, 2)
        })
    return pd.DataFrame(rows)