# import numpy as np

# participants = {
#     0: list(range(2914, 2951)),
#     1: list(range(2871, 2904)),
#     2: list(range(2323, 2356)),
#     3: list(range(2285, 2314)),
#     4: list(range(1646, 1675)),
#     5: list(range(1503, 1535)) + list(range(1537, 1544)),
# }

# def filter_participants(feats, dataset):
#   return [
#     filter_by_id(feats, dataset, participants[i])
#     for i in range(6)
#   ]


# def filter_targets(dataset):
#     return [
#       np.array([dataset.targets[dataset.img_ids.index(img_id)] for img_id in participants[i]])
#       for i in range(6)
#     ]

# def filter_by_id(feats, dataset, img_ids):
#   indices = [dataset.img_ids.index(img_id) for img_id in img_ids]
#   filtered_feats = {}
#   for model_name, layers in feats.items():
#     filtered_feats[model_name] = {} 
#     for layer_name, feat in layers.items():
#       filtered_feats[model_name][layer_name] = feat[indices]
#   return filtered_feats


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
    img_id_to_idx = {img_id: idx for idx, img_id in enumerate(dataset.img_ids)}
    participant_indices = {
        part_id: [img_id_to_idx[img_id] for img_id in img_ids]
        for part_id, img_ids in participants.items()
    }
    partitioned_feats = {part_id: {} for part_id in participants}
    for model_name in list(feats.keys()):
        for part_id in participants:
            partitioned_feats[part_id][model_name] = {}
        for layer_name in list(feats[model_name].keys()):
            layer_feat = feats[model_name][layer_name]
            for part_id, indices in participant_indices.items():
                partitioned_feats[part_id][model_name][layer_name] = layer_feat[indices]
            del feats[model_name][layer_name]
        del feats[model_name]

    return partitioned_feats

def filter_targets(dataset):
    img_id_to_idx = {img_id: idx for idx, img_id in enumerate(dataset.img_ids)}
    return [
        np.array([dataset.targets[img_id_to_idx[img_id]] for img_id in participants[i]])
        for i in range(6)
    ]
