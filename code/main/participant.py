from feature import filter_by_id
import numpy as np

participants = {
    0: list(range(2914, 2951)),
    1: list(range(2871, 2904)),
    2: list(range(2323, 2356)),
    3: list(range(2285, 2314)),
    4: list(range(1646, 1675)),
    5: list(range(1503, 1535)) + list(range(1537, 1544)),
}

def filter_participants(feats, dataset):
  return [
    filter_by_id(feats, dataset, participants[i])
    for i in range(6)
  ]


def filter_targets(dataset):
    return [
        np.array([dataset.targets[dataset.img_ids.index(img_id)] for img_id in participants[i]])
        for i in range(6)
    ]
