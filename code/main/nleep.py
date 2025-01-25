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