# return_nodes are PyTorch specific and can be considered by model examination

MODEL_ZOO = {
  'vgg19': {
    'input_dim': (224, 224), 
    'return_nodes': ['features.34', 'features.35', 'avgpool', 'classifier.6'],
    # note that avgpool layer has been provided for vgg19 in PyTorch
    # which doesn't exist in the original implementation (nor, say, in Keras).
    # AdaptiveAvgPool2d((7,7)) (which isn't a GAP layer) was added for better flexibility
    # and is not directly comparable to other architectures (at least by dimensionality)
    # We still consider additional avgpool layer in experiments for vgg19
  },
  'resnet50': {
    'input_dim': (224, 224), 
    'return_nodes': ['layer4.2.relu_2', 'avgpool', 'fc'],
  },
  'inception_v3': {
    'input_dim': (299, 299), 
    'return_nodes': ['Mixed_7c.branch_pool.conv', 'avgpool', 'fc'],
  },
  'efficientnet_b0': {
    'input_dim': (224, 224), 
    'return_nodes': ['features.8', 'avgpool', 'classifier.1'],
  },
  'efficientnet_b1': {
    'input_dim': (240, 240), 
    'return_nodes': ['features.8', 'avgpool', 'classifier.1'],
  },
  'efficientnet_b6': {
    'input_dim': (528, 528), 
    'return_nodes': ['features.8', 'avgpool', 'classifier.1'],
  },
}


# Parameters noted in the paper
FULL_PARAM_GRID = {
  'umap': {
    'n_neighbors': [5, 10, 15, 30],
    'min_dist': [0.01, 0.05, 0.1, 0.3],
    'metric': ["euclidean", "cosine"]
  },
  'tsne': {
    'perplexity': [10, 15, 20, 25],
    'learning_rate': [100, 300, 600, 1000],
    'metric': ["euclidean", "cosine"]
  },
  'pca': {
    'whiten': [True, False],
    'svd_solver': ['full', 'auto'],
  }
}

# Minimal set of parameters for quicker experiments
MIN_PARAM_GRID = {
  'umap': {
    'n_neighbors': [5, 15, 30],
    'min_dist': [0.01, 0.1, 0.3],
    'metric': ["euclidean"]
  },
  'tsne': {
    'perplexity': [10, 15, 25],
    'learning_rate': [100],
    'metric': ["euclidean"]
  },
  'pca': {
    'whiten': [False],
    'svd_solver': ['full'],
  }
}
