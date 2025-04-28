# Evaluation of spatial feature extraction methods in facial and gesture detection problems

Code repository for the experiments conducted in the paper. 

We explore the effectiveness of _spatial feature extraction_ methods in the context of _hand gesture classification_ tasks, proposing a methodology for assessing the transferability of various pre-trained models to a target domain. A major concern is the computational cost and transferability of these models. We evaluate key _model transferability estimation (MTE)_ methods and propose a multi-step, resource-efficient methodology to qualify the suitability of pretrained learning models for pattern recognition.

**Keywords:** _model transferability estimation (MTE), transfer learning, hand gesture recognition, sign language, pre-trained model comparison, Explainable CNN_

## Usage

It is recommended to run the project on _Google Colab_ to avoid local setup issues.  
The required packages are listed in `requirements.txt`. All main experiments can be replicated by running `experiments.ipynb`. Before running, ensure that the folder paths are correctly assigned. In case of storing and loading generated features and embeddings, the project expects an `out/` directory with two subfolders: `feats/` and `embeddings/`.

## Ackowledgements

The authors would like to thank the members of the Azerbaijan Deaf Society—Samir Sadigov and Shamsi Shahbazova, and the sign language translators—Konul Guliyeva, Shamil Sabirzade, and Heyder Rahimov for their guidance, support, and contribution.