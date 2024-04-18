import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE
import re
from sklearn.cluster import KMeans
from collections import defaultdict
import os


def test_tsne(conf, num_clusters=5, plot=False, plot_imgs=False, cluster_info=True):
  last_conv_outputs = conf.get_last_conv_outputs()
  tsne_embedding = get_tsne(last_conv_outputs, plot=plot)
  clusters = cluster(tsne_embedding, num_clusters=num_clusters)
  if plot_imgs:
    visualize_clusters(tsne_embedding, conf.imgs, clusters)
  if cluster_info:
    get_cluster_info(clusters, conf.img_paths)


def get_tsne(last_conv_outputs, plot=True):
    tsne = TSNE(n_components=2, random_state=42)
    tsne_embedding = tsne.fit_transform(last_conv_outputs)

    if plot:
        plt.scatter(tsne_embedding[:, 0], tsne_embedding[:, 1])

        for i in range(len(tsne_embedding)):
          plt.annotate(str(i), (tsne_embedding[i, 0], tsne_embedding[i, 1]),  fontsize=8)

        plt.title('t-SNE Embedding of Last Convolutional Layer Features')
        plt.show()

    return tsne_embedding



def cluster(tsne_embedding, num_clusters=6, plot=True):
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    clusters = kmeans.fit_predict(tsne_embedding)

    if plot:
        # Plot clusters onto t-SNE embedding
        plt.scatter(tsne_embedding[:, 0], tsne_embedding[:, 1], c=clusters, cmap='viridis', alpha=0.5)
        plt.title(f't-SNE Embedding with {num_clusters} K-Means Clusters')
        plt.colorbar()

        # Annotate each cluster
        for i in range(num_clusters):
            cluster_center = kmeans.cluster_centers_[i]
            plt.annotate(f'{i + 1}', cluster_center, fontsize=8, ha='center', va='center', color='red')
            
        plt.show()

    return clusters

def visualize_clusters(tsne_embedding, imgs, clusters, max_images_per_line=6):
    num_clusters = len(np.unique(clusters))
    
    plt.figure(figsize=(15, 5 * num_clusters))

    for cluster_id in range(num_clusters):
        cluster_indices = np.where(clusters == cluster_id)[0]
        num_images = len(cluster_indices)

        num_cols = min(num_images, max_images_per_line)
        num_rows = int(np.ceil(num_images / num_cols))

        for i, idx in enumerate(cluster_indices, start=1):
            plt.subplot(num_rows, num_cols, i)
            plt.imshow(imgs[idx])
            # plt.title(f'Cluster {cluster_id + 1}\nImage {idx}', fontsize=8)
            plt.axis('off')

        plt.show()

    plt.show()


def get_cluster_info(clusters, img_paths, img_names=False):
    num_clusters = len(np.unique(clusters))

    for cluster_id in range(num_clusters):
        cluster_indices = np.where(clusters == cluster_id)[0]
        
        print(f'Cluster {cluster_id + 1}: ', end='')

        letter_counts = defaultdict(int)

        for i, idx in enumerate(cluster_indices):
            img_path = img_paths[idx]
            relative_path = os.path.sep.join(img_path.split(os.path.sep)[-2:])

            letter = relative_path.split(os.path.sep)[0]
            letter_counts[letter] += 1

            if img_names:
                print(relative_path, end=', ')

        total_count = len(cluster_indices)
        print(f'\nLetter Counts: {dict(letter_counts)}')
        print(f'Total Count: {total_count}\n')
