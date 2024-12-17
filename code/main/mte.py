from loader import ModelLoader
from feature import get_f_y_pred
import torch

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