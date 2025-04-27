from typing import Dict
import torch
from config import MODEL_ZOO
from utils import get_fc_tensor, predict
import numpy as np
from tllib.ranking.hscore import regularized_h_score
from tllib.ranking.transrate import transrate
from tllib.ranking.logme import log_maximum_evidence
from tllib.ranking.leep import log_expected_empirical_prediction
from sklearn.mixture import GaussianMixture

def get_mte_scores(
    feats: Dict[str, torch.Tensor],
    targets: torch.Tensor,
) -> None:
    model_scores = {
        "LEEP": {},
        "LogME": {},
        "Regularized H-Score": {},
        "Transrate": {}
    }

    for model_name in MODEL_ZOO:
        print(f'Calculating scores for {model_name}...')
        preds, _, feat_tensor = predict(model_name, feats)
        f = feat_tensor.numpy()
        y = np.array(targets)

        model_scores["LEEP"][model_name] = log_expected_empirical_prediction(preds.numpy(), y)
        model_scores["LogME"][model_name] = log_maximum_evidence(f, y)
        model_scores["Regularized H-Score"][model_name] = regularized_h_score(f, y)
        model_scores["Transrate"][model_name] = transrate(f, y)

    ranked_scores = {}
    for metric in model_scores:
        ranked_scores[metric] = sorted(model_scores[metric].items(), key=lambda x: x[1], reverse=True)

    for metric, models in ranked_scores.items():
        print(f"\n=== Ranked Models by {metric} ===")
        print("{:<5} {:<20} {:<15}".format("Rank", "Model", "Score"))
        print("-" * 45)
        for rank, (model, score) in enumerate(models, start=1):
            print("{:<5} {:<20} {:.6f}".format(rank, model, score))
        print()


def get_nleep_score(cluster, random_state=None, verbose=True):
  'see cluster.py'
  n_components = cluster.embedding.n_clusters
  embedding = cluster.embedding.reducer.embedding
  targets = cluster.embedding.reducer.targets

  if verbose:
    print(f'Generating NLEEP for the best cluster\n{cluster}')
  
  gmm = GaussianMixture(n_components=n_components, random_state=random_state)
  gmm.fit(embedding)
  gmm_predictions = gmm.predict_proba(embedding)

  cluster.nleep_score = log_expected_empirical_prediction(gmm_predictions, targets)
