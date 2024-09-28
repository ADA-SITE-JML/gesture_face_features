import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from collections import defaultdict
import os
import pickle

class TSNECluster:
    def __init__(self, config, perplexity=30):
        self.config = config
        self.num_clusters = len(config.letter)
        self.perplexity = perplexity
        self.tsne_embedding = None
        self.clusters = None
        self.letter_marker_map = {
            'A': 'o', 'H': 's', 
            'L': '^', 'N': 'v', 
            'O': 'P', 'P': 'X', 'R': '*'
        }
        self.colors = plt.cm.rainbow(np.linspace(0, 1, self.num_clusters))


    def run(self, plot=True, save_data=True):
        self.tsne_embedding = self.get_tsne()
        self.clusters = self.get_clusters()
        if plot:
            self.visualize()
        if save_data:
            self.save()


    def get_tsne(self):
        outputs = self.config.get_last_dense_outputs()
        tsne = TSNE(n_components=2, random_state=42, perplexity=self.perplexity)
        tsne_embedding = tsne.fit_transform(outputs)
        return tsne_embedding


    def get_clusters(self):
        kmeans = KMeans(n_clusters=self.num_clusters, random_state=42)
        clusters = kmeans.fit_predict(self.tsne_embedding)
        return clusters
        

    def visualize(self):
        fig, ax = plt.subplots(figsize=(8, 5))
        handles = [] # for legend

        for cluster_id in range(self.num_clusters):
            cluster_indices = np.where(self.clusters == cluster_id)[0]
            cluster_letters = [self.config.img_paths[idx].split(os.path.sep)[-2][0] for idx in cluster_indices]

            for letter in set(self.config.letter):
                letter_indices = [i for i, l in enumerate(cluster_letters) if l == letter]
                count = len(letter_indices) if letter_indices else 0
                handle = ax.scatter(
                    self.tsne_embedding[cluster_indices[letter_indices], 0],
                    self.tsne_embedding[cluster_indices[letter_indices], 1],
                    color=self.colors[cluster_id],
                    marker=self.letter_marker_map[letter],
                    label=f'{letter}: {count}'
                )
                handles.append(handle)

       
        # ax.legend(handles=handles, loc='center', bbox_to_anchor=(0.5, -0.2), ncol=num_clusters)
        # ax.set_title(f"t-SNE Visualization for {self.config.model_name}")
        ax.set_xticks([])
        ax.set_yticks([])

        plt.show()

    def get_info(self):
        letter_marker_map = {
          'A': '▲',  # Triangle
          'H': '■',  # Square
          'L': '▼',  # Inverted triangle
          'N': '■',  # Square (same as 'H')
          'O': '●',  # Circle
          'P': '★',  # Star
          'R': '✚'   # Cross
        }

        cluster_counts = defaultdict(lambda: defaultdict(int))
        total_counts = defaultdict(int)
        for cluster_id in range(self.num_clusters):
            cluster_indices = np.where(self.clusters == cluster_id)[0]
            cluster_letters = [self.config.img_paths[idx].split(os.path.sep)[-2][0] for idx in cluster_indices]

            for letter in set(self.config.letter):
                letter_indices = [i for i, l in enumerate(cluster_letters) if l == letter]
                count = len(letter_indices) if letter_indices else 0
                cluster_counts[letter][cluster_id] = count
                total_counts[cluster_id] += count

        fig, ax = plt.subplots(figsize=(12, 4))
        ax.set_axis_off()
        cell_text = []
        row_labels = []
        for row_label, row_data in cluster_counts.items():
            row_text = [f"${letter_marker_map[row_label]}$ {row_label}"]
            for col in range(self.num_clusters):
                count = row_data[col]
                row_text.append(count)
            cell_text.append(row_text)
            row_labels.append(row_label)

        total_row = ['Total']
        for col in range(self.num_clusters):
            total_row.append(total_counts[col])
        cell_text.append(total_row)

        col_labels = ['Letters'] + [f'Cluster {i+1}' for i in range(self.num_clusters)]
        n_rows = len(cell_text)

        table = ax.table(cellText=cell_text, colLabels=col_labels, loc='center', cellLoc='center', cellColours=[['white'] * (self.num_clusters + 1)] * (n_rows - 1) + [['lightgray'] + ['white'] * self.num_clusters])

        table[0, 0].set_facecolor('lightgray')
        for j in range(1, self.num_clusters + 1):
            table[0, j].set_facecolor(self.colors[j-1])
            table[n_rows, j].set_facecolor(self.colors[j-1])

        plt.show()






    def save(self):
        data = {
            'tsne_embedding': self.tsne_embedding,
            'clusters': self.clusters,
            # 'config': self.config
        }
        filename = os.path.join(self.config.code_path, f"{self.config.model_name}_tsne.pkl")
        with open(filename, 'wb') as f:
            pickle.dump(data, f)
        print(f"Data saved to {filename}")


    def load(self, filename):
        filename = os.path.join(self.config.code_path, filename)
        with open(filename, 'rb') as f:
            data = pickle.load(f)
        self.tsne_embedding = data['tsne_embedding']
        self.clusters = data['clusters']
        # self.config = data['config']
        # self.num_clusters = len(self.config.letter)
        print(f"Data loaded from {filename}")
