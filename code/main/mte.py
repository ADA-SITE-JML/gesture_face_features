from loader import ModelLoader
from feature import get_f_y_pred
import torch


# coding=utf-8
# Copyright 2024 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Implementation of NLEEP (Ranking Neural Checkpoints).

Li, Yandong, et al. "Ranking neural checkpoints." Proceedings of the IEEE/CVF
Conference on Computer Vision and Pattern Recognition. 2021.
https://openaccess.thecvf.com/content/CVPR2021/papers/Li_Ranking_Neural_Checkpoints_CVPR_2021_paper.pdf
"""


# coding=utf-8
# Copyright 2024 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Implementation of Log Expected Empirical Prediction(LEEP).

Nguyen, Cuong, et al. "Leep: A new measure to evaluate transferability of
learned representations." International Conference on Machine Learning. PMLR,
2020. https://arxiv.org/abs/2002.12462
"""


from sklearn.mixture import GaussianMixture
import tensorflow as tf


def get_leep_score(predictions, target_labels):
  """Implementation of Log Expected Empirical Prediction(LEEP).

  Args:
    predictions: matrix [N, S] of source predictions obtained from the target
      data, where N is the number of datapoints and S the source classes number.
    target_labels: ground truth target labels of dimension [N, 1].

  Returns:
    leep: transferability metric score.

  """

  num_target_classes = tf.reduce_max(target_labels) + 1
  one_hot_encoding_t = tf.squeeze(tf.one_hot(
      tf.cast(target_labels, tf.int32), depth=num_target_classes))

  occurrences_s_t = tf.einsum('bt,bs->st', one_hot_encoding_t, predictions)
  occurrences_s = tf.reduce_sum(occurrences_s_t, axis=-1, keepdims=True)
  probability_t_given_s = tf.math.divide_no_nan(occurrences_s_t, occurrences_s)

  target_predictions = tf.matmul(predictions, probability_t_given_s)
  eep = tf.reduce_sum(target_predictions * one_hot_encoding_t, axis=-1)
  # If a prediction is missing, we assume random (uniform) predictions.
  leep = tf.reduce_mean(tf.where(
      eep > 0, tf.math.log(eep), -tf.math.log(float(num_target_classes))))

  return leep


def get_nleep_score(
    features,
    target_labels,
    num_components_gmm,
    random_state=123):
  """Computes NLEEP (Ranking Neural Checkpoints).

  Args:
    features: matrix [N, D] of source features obtained from the target data,
      where N is the number of datapoints and D their dimensionionality.
    target_labels: ground truth target labels of dimension [N, 1].
    num_components_gmm: gaussian components (5*target class number in the paper)
    random_state: random seed for the GaussianMixture initialization.

  Returns:
    nleep: transferability metric score.

  """
  gmm = GaussianMixture(
      n_components=num_components_gmm, random_state=random_state).fit(features)
  gmm_predictions = gmm.predict_proba(features)
  nleep = get_leep_score(gmm_predictions.astype('float32'), target_labels)
  return nleep








# based on https://github.com/zhangyikaii/Model-Spider/blob/main/tools/feature_extractor.py

available_methods = [
    'LogME',
    'NCE',
    'LEEP',
    'H_Score',
    'NLEEP',
    # 'OTCE',
    'PACTranDirichlet',
    'PACTranGamma',
    'GBC',
    # 'DEPARA',
    'LFC',
    'ZERO'
]

def score_MTE(feats, targets, model_pool, method):
  scores = {}

  for model_name in model_pool:
    f, y, predictions, idx = get_f_y_pred(model_name, feats, targets)

    if method == 'LogME':
        from model_spider.mptms.LogME import LogME
        logme = LogME(regression=False)
        score = logme.fit(f, y)
    elif method == 'NCE':
        from  model_spider.mptms.NCE import NCE
        score = NCE(source_label=torch.argmax(predictions, dim=1).numpy(), target_label=y)
    elif method == 'LEEP':
        from  model_spider.mptms.LEEP import LEEP
        score = LEEP(prob_np_all=predictions.numpy(), label_np_all=y)
    elif method == 'H_Score':
        from  model_spider.mptms.H_Score import H_Score
        score = H_Score(features=f, labels=y)
    elif method == 'NLEEP':
        from  model_spider.mptms.LEEP import NLEEP
        score = NLEEP(features_np_all=f, label_np_all=y)
    # elif do_k == 'OTCE':
    #     from  model_spider.mptms.OTCE import OTCE
    #     imagenet_features = np.load(f'{feat_path}/{model_name}_ImageNet5Shot_features.npy')
    #     imagenet_targets = np.load(f'{feat_path}/ImageNet5Shot_targets.npy')
    #     score = OTCE(src_x=torch.tensor(imagenet_features, dtype=torch.float),
    #                   tar_x=feat_tensor,
    #                   src_y=imagenet_targets, tar_y=y)
    elif method == 'PACTranDirichlet':
        from model_spider.mptms.PACTran import PACTranDirichlet
        score = -PACTranDirichlet(prob_np_all=predictions.numpy(), label_np_all=y, alpha=1.)
    elif method == 'PACTranGamma':
        from model_spider.mptms.PACTran import PACTranGamma
        score = -PACTranGamma(prob_np_all=predictions.numpy(), label_np_all=y, alpha=1.)
    elif method == 'GBC':
        from  model_spider.mptms.GBC import GBC
        score = GBC(features=f, labels=y)
    # elif do_k == 'DEPARA':
    #     from  model_spider.mptms.DEPARA import DEPARA
    #     imagenet_features = np.load(f'{feat_path}/{model_name}_ImageNet5Shot_features.npy')
    #     score = DEPARA(feature_p=imagenet_features, feature_q=f)
    elif method == 'LFC':
        from  model_spider.mptms.LFC import LFC
        score = LFC(torch.tensor(f), torch.tensor(y))
    elif method == 'ZERO':
        score = 0
    else:
        raise Exception(f'Unknown evaluation method: {method}')

    scores[model_name] = score
    print(f'{method} score for {model_name}: {score}')

  return scores



import warnings

import numpy as np
from numba import njit


@njit
def each_evidence(y_, f, fh, v, s, vh, N, D):
    """
    compute the maximum evidence for each class
    """
    epsilon = 1e-5
    alpha = 1.0
    beta = 1.0
    lam = alpha / beta
    tmp = (vh @ (f @ np.ascontiguousarray(y_)))
    for _ in range(11):
        # should converge after at most 10 steps
        # typically converge after two or three steps
        gamma = (s / (s + lam)).sum()
        # A = v @ np.diag(alpha + beta * s) @ v.transpose() # no need to compute A
        # A_inv = v @ np.diag(1.0 / (alpha + beta * s)) @ v.transpose() # no need to compute A_inv
        m = v @ (tmp * beta / (alpha + beta * s))
        alpha_de = (m * m).sum()
        alpha = gamma / (alpha_de + epsilon)
        beta_de = ((y_ - fh @ m) ** 2).sum()
        beta = (N - gamma) / (beta_de + epsilon)
        new_lam = alpha / beta
        if np.abs(new_lam - lam) / lam < 0.01:
            break
        lam = new_lam
    evidence = D / 2.0 * np.log(alpha) \
               + N / 2.0 * np.log(beta) \
               - 0.5 * np.sum(np.log(alpha + beta * s)) \
               - beta / 2.0 * (beta_de + epsilon) \
               - alpha / 2.0 * (alpha_de + epsilon) \
               - N / 2.0 * np.log(2 * np.pi)
    return evidence / N, alpha, beta, m


# use pseudo data to compile the function
# D = 20, N = 50
f_tmp = np.random.randn(20, 50).astype(np.float64)
each_evidence(np.random.randint(0, 2, 50).astype(np.float64), f_tmp, f_tmp.transpose(), np.eye(20, dtype=np.float64), np.ones(20, dtype=np.float64), np.eye(20, dtype=np.float64), 50, 20)


@njit
def truncated_svd(x):
    u, s, vh = np.linalg.svd(x.transpose() @ x)
    s = np.sqrt(s)
    u_times_sigma = x @ vh.transpose()
    k = np.sum((s > 1e-10) * 1)  # rank of f
    s = s.reshape(-1, 1)
    s = s[:k]
    vh = vh[:k]
    u = u_times_sigma[:, :k] / s.reshape(1, -1)
    return u, s, vh
truncated_svd(np.random.randn(20, 10).astype(np.float64))


class LogME(object):
    def __init__(self, regression=False):
        """
            :param regression: whether regression
        """
        self.regression = regression
        self.fitted = False
        self.reset()

    def reset(self):
        self.num_dim = 0
        self.alphas = []  # alpha for each class / dimension
        self.betas = []  # beta for each class / dimension
        # self.ms.shape --> [C, D]
        self.ms = []  # m for each class / dimension

    def _fit_icml(self, f: np.ndarray, y: np.ndarray):
        """
        LogME calculation proposed in the ICML 2021 paper
        "LogME: Practical Assessment of Pre-trained Models for Transfer Learning"
        at http://proceedings.mlr.press/v139/you21b.html
        """
        fh = f
        f = f.transpose()
        D, N = f.shape
        v, s, vh = np.linalg.svd(f @ fh, full_matrices=True)

        evidences = []
        self.num_dim = y.shape[1] if self.regression else int(y.max() + 1)
        for i in range(self.num_dim):
            y_ = y[:, i] if self.regression else (y == i).astype(np.float64)
            evidence, alpha, beta, m = each_evidence(y_, f, fh, v, s, vh, N, D)
            evidences.append(evidence)
            self.alphas.append(alpha)
            self.betas.append(beta)
            self.ms.append(m)
        self.ms = np.stack(self.ms)
        return np.mean(evidences)

    def _fit_fixed_point(self, f: np.ndarray, y: np.ndarray):
        """
        LogME calculation proposed in the arxiv 2021 paper
        "Ranking and Tuning Pre-trained Models: A New Paradigm of Exploiting Model Hubs"
        at https://arxiv.org/abs/2110.10545
        """
        N, D = f.shape  # k = min(N, D)
        if N > D: # direct SVD may be expensive
            u, s, vh = truncated_svd(f)
        else:
            u, s, vh = np.linalg.svd(f, full_matrices=False)
        # u.shape = N x k
        # s.shape = k
        # vh.shape = k x D
        s = s.reshape(-1, 1)
        sigma = (s ** 2)

        evidences = []
        self.num_dim = y.shape[1] if self.regression else int(y.max() + 1)
        for i in range(self.num_dim):
            y_ = y[:, i] if self.regression else (y == i).astype(np.float64)
            y_ = y_.reshape(-1, 1)
            x = u.T @ y_  # x has shape [k, 1], but actually x should have shape [N, 1]
            x2 = x ** 2
            res_x2 = (y_ ** 2).sum() - x2.sum()  # if k < N, we compute sum of xi for 0 singular values directly

            alpha, beta = 1.0, 1.0
            for _ in range(11):
                t = alpha / beta
                gamma = (sigma / (sigma + t)).sum()
                m2 = (sigma * x2 / ((t + sigma) ** 2)).sum()
                res2 = (x2 / ((1 + sigma / t) ** 2)).sum() + res_x2
                alpha = gamma / (m2 + 1e-5)
                beta = (N - gamma) / (res2 + 1e-5)
                t_ = alpha / beta
                evidence = D / 2.0 * np.log(alpha) \
                           + N / 2.0 * np.log(beta) \
                           - 0.5 * np.sum(np.log(alpha + beta * sigma)) \
                           - beta / 2.0 * res2 \
                           - alpha / 2.0 * m2 \
                           - N / 2.0 * np.log(2 * np.pi)
                evidence /= N
                if abs(t_ - t) / t <= 1e-3:  # abs(t_ - t) <= 1e-5 or abs(1 / t_ - 1 / t) <= 1e-5:
                    break
            evidence = D / 2.0 * np.log(alpha) \
                       + N / 2.0 * np.log(beta) \
                       - 0.5 * np.sum(np.log(alpha + beta * sigma)) \
                       - beta / 2.0 * res2 \
                       - alpha / 2.0 * m2 \
                       - N / 2.0 * np.log(2 * np.pi)
            evidence /= N
            m = 1.0 / (t + sigma) * s * x
            m = (vh.T @ m).reshape(-1)
            evidences.append(evidence)
            self.alphas.append(alpha)
            self.betas.append(beta)
            self.ms.append(m)
        self.ms = np.stack(self.ms)
        return np.mean(evidences)

    _fit = _fit_fixed_point

    def fit(self, f: np.ndarray, y: np.ndarray):
        """
        :param f: [N, F], feature matrix from pre-trained model
        :param y: target labels.
            For classification, y has shape [N] with element in [0, C_t).
            For regression, y has shape [N, C] with C regression-labels

        :return: LogME score (how well f can fit y directly)
        """
        if self.fitted:
            warnings.warn('re-fitting for new data. old parameters cleared.')
            self.reset()
        else:
            self.fitted = True
        f = f.astype(np.float64)
        if self.regression:
            y = y.astype(np.float64)
            if len(y.shape) == 1:
                y = y.reshape(-1, 1)
        return self._fit(f, y)

    def predict(self, f: np.ndarray):
        """
        :param f: [N, F], feature matrix
        :return: prediction, return shape [N, X]
        """
        if not self.fitted:
            raise RuntimeError("not fitted, please call fit first")
        f = f.astype(np.float64)
        logits = f @ self.ms.T
        if self.regression:
            return logits
        return np.argmax(logits, axis=-1)