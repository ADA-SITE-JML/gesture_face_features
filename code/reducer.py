import os
import numpy as np
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
import umap
from sklearn.model_selection import ParameterGrid
import matplotlib.pyplot as plt


class EmbeddingProcessor:
    def __init__(self, partic_feats, labels, img_numbers, partic_num,  embedding_path=None, method="umap", 
                 param_grid=None, n_clusters=7, n_components=2, verbose=False, random_state=None, recompute=False):
        self.partic_feats = partic_feats
        self.labels = labels
        self.img_numbers = img_numbers
        self.partic_num = partic_num
        self.method = method
        self.param_grid = param_grid
        self.n_clusters = n_clusters
        self.n_components = n_components
        self.path = embedding_path
        self.verbose = verbose
        self.random_state = random_state
        self.recompute = recompute
        self.embeddings = {}
        self.results_info = {}

    def _load_or_compute_embeddings(self):
        """
        Load pre-computed embeddings or compute them if not found.
        """
        if self.path and not self.recompute:
          file_name = f"{self.method}_embeddings_participant_{self.partic_num}_rs_{self.random_state}.npy"
          file_path = os.path.join(self.path, file_name)

          if os.path.exists(file_path):
              data = np.load(file_path, allow_pickle=True).item()
              self.embeddings = data['embeddings']
              self.results_info = data['results_info']
              print(f"Loaded embeddings and results from {file_path}.")
              return self.embeddings, self.results_info

        if self.method == "umap":
            self.param_grid = self.param_grid or {
              'n_neighbors': [5, 10, 15, 30, 50],
              'min_dist': [0.01, 0.05, 0.1, 0.3],
              'metric': ["euclidean", "cosine"]
            }
        elif self.method == "tsne":
            self.param_grid = self.param_grid or {
                'perplexity': [10, 15, 20, 25],
                'learning_rate': [100, 300, 600, 1000],
                'metric': ["euclidean", "cosine"]
            }
        else:
            raise ValueError("Method must be 'umap' or 'tsne'")

        for model_name, layers in self.partic_feats.items():
            self.embeddings[model_name] = {}
            self.results_info[model_name] = {}

            for layer_name, features in layers.items():
                best_embedding = None
                best_score = -np.inf
                best_params = None

                for params in ParameterGrid(self.param_grid):
                    embedding = self._compute_embedding(features, params)
                    clusters = self._calculate_clusters(embedding)
                    layer_score = self._calculate_similarity_score(clusters)

                    self.results_info[model_name].setdefault(layer_name, []).append({'params': params, 'score': layer_score})

                    if layer_score > best_score:
                        best_score = layer_score
                        best_embedding = embedding
                        best_params = params

                self.embeddings[model_name][layer_name] = best_embedding
                print(f"Best params for {model_name} layer {layer_name}: {best_params} with score: {best_score:.2f}")
        
        if self.path:
          np.save(file_path, {'embeddings': self.embeddings, 'results_info': self.results_info})
          print(f"Saved embeddings and results to {file_path}.")

        return self.embeddings, self.results_info

    def _compute_embedding(self, features, params):
        """
        Compute the embedding using UMAP or t-SNE based on the provided parameters.
        """
        if self.method == "umap":
            reducer = umap.UMAP(
                n_neighbors=params['n_neighbors'],
                min_dist=params['min_dist'],
                n_components=self.n_components,
                metric=params['metric'],
                random_state=self.random_state
            )
        elif self.method == "tsne":
            reducer = TSNE(
                n_components=self.n_components,
                perplexity=params['perplexity'],
                learning_rate=params['learning_rate'],
                metric=params['metric'],
                random_state=self.random_state
            )

        if len(features.shape) > 2:
            # print(f"Flattening features of shape {features.shape}")
            features = features.reshape(features.shape[0], -1)

        return reducer.fit_transform(features)

    def _calculate_clusters(self, embedding):
        """
        Use KMeans to cluster the embeddings.
        """
        kmeans = KMeans(n_clusters=self.n_clusters, random_state=self.random_state)
        cluster_labels = kmeans.fit_predict(embedding)
        return self._group_images_by_cluster(cluster_labels)

    def _group_images_by_cluster(self, cluster_labels):
        """
        Group images by their assigned cluster.
        """
        clusters = {i: [] for i in range(self.n_clusters)}
        for img_num, cluster, label in zip(self.img_numbers, cluster_labels, self.labels):
            clusters[cluster].append((img_num, label))
        return clusters

    def _calculate_similarity_score(self, clusters):
        """
        Calculate the similarity score based on predefined letter pair similarity scores.
        """
        similarity_scores = {
            ('R', 'N'): 0.8, ('R', 'O'): 0.6, ('N', 'O'): 0.6,
            ('H', 'P'): 0.8, ('P', 'L'): 0.8, ('H', 'L'): 0.6,
        }
        for letter in 'AHLNPRO':
            similarity_scores[(letter, letter)] = 1.0

        def get_similarity_score(letter1, letter2):
            return similarity_scores.get((letter1, letter2)) or similarity_scores.get((letter2, letter1), 0)

        total_score = 0
        total_pairs = 0

        for cluster_id, images in clusters.items():
            letters = [label for _, label in images]
            cluster_score = 0
            num_pairs = 0

            letter_counts = {letter: letters.count(letter) for letter in set(letters)}

            for letter1, count1 in letter_counts.items():
                for letter2, count2 in letter_counts.items():
                    if letter1 != letter2:
                        pair_score = get_similarity_score(letter1, letter2)
                        cluster_score += pair_score * count1 * count2
                        num_pairs += count1 * count2
                    else:
                        pair_score = get_similarity_score(letter1, letter2)
                        cluster_score += pair_score * (count1 * (count1 - 1) // 2)
                        num_pairs += count1 * (count1 - 1) // 2

            cluster_score = cluster_score / num_pairs if num_pairs > 0 else 0
            total_score += cluster_score
            total_pairs += 1

        return total_score / total_pairs if total_pairs > 0 else 0

    def analyze_results(self, top_n=5):
        """
        Analyze the results based on the top N layers.
        """
        analysis = {}
        overall_best_score = float('-inf')  # Initialize to negative infinity
        overall_best_info = None  # To store the best score details

        for model_name, layers in self.results_info.items():
            analysis[model_name] = {}

            all_scores = []
            for layer_name, results in layers.items():
                all_scores.extend([(model_name, layer_name, res['params'], res['score']) for res in results])

            all_scores.sort(key=lambda x: x[3], reverse=True)  # Sort by score in descending order

            # Select the top N best layers
            analysis[model_name]['top_n_best_layers'] = all_scores[:top_n]

            if all_scores:
                top_n_scores = [layer[3] for layer in analysis[model_name]['top_n_best_layers']]
                average_top_n_score = sum(top_n_scores) / len(top_n_scores) if top_n_scores else 0
                analysis[model_name]['average_top_n_score'] = average_top_n_score

                # Identify the best model based on score
                best_model = max(all_scores, key=lambda x: x[3])  # Best model based on score
                analysis[model_name]['best_model'] = {
                    'model': best_model[0],
                    'layer': best_model[1],
                    'params': best_model[2],
                    'score': best_model[3]
                }

                # Check for overall best score
                if best_model[3] > overall_best_score:
                    overall_best_score = best_model[3]
                    overall_best_info = {
                        'model': best_model[0],
                        'layer': best_model[1],
                        'params': best_model[2],
                        'score': best_model[3]
                    }

        # Print the analysis results
        print("\nAnalysis Results:")
        for model_name, data in analysis.items():
            print(f"\nModel: {model_name}")
            print(f"Average Score of Top {top_n} Layers: {data['average_top_n_score']:.4f}")
            print(f"Top {top_n} Best Layers:")
            for layer_info in data['top_n_best_layers']:
                print(f"  Layer: {layer_info[1]}, Score: {layer_info[3]:.2f}, Params: {layer_info[2]}")

        print()
        print(f"Best score: {overall_best_info}")
    
    def plot_image_clusters(self, clustered_images, model_name, layer_name, target_size=(100, 100)):
        """
        Plot the clustered images.
        """
        clusters = clustered_images.get(model_name, {}).get(layer_name, {})

        if not clusters:
            print(f"No clusters found for {model_name} {layer_name}.")
            return

        # Determine the maximum number of images in any cluster for consistent row widths
        max_images_per_row = max(len(img_list) for img_list in clusters.values())
        fig, axs = plt.subplots(len(clusters), max_images_per_row, figsize=(max_images_per_row * 3, len(clusters) * 3))

        for row_idx, (cluster, img_list) in enumerate(clusters.items()):
            for col_idx in range(max_images_per_row):
                if col_idx < len(img_list):
                    img_num, label = img_list[col_idx]

                    if img_num in self.img_numbers:
                        img_index = self.img_numbers.index(img_num)
                        img = imgs[img_index]  # Get the image from imgs list

                        # Resize the image
                        resized_img = img.resize(target_size)

                        # Plot the image
                        axs[row_idx, col_idx].imshow(resized_img)
                        axs[row_idx, col_idx].axis('off')  # Hide axes
                        axs[row_idx, col_idx].set_title(f'Letter: {label}')  # Optional title
                    else:
                        print(f"Warning: img_num {img_num} is out of bounds for imgs list.")
                        axs[row_idx, col_idx].axis('off')  # Hide unused subplots
                else:
                    axs[row_idx, col_idx].axis('off')  # Hide unused subplots

        # Adjust layout for the entire figure
        plt.tight_layout()
        plt.show()

    def visualize_embedding(self, embedding, labels, cluster_labels, img_numbers, model_name, layer_name, n_clusters=7):
        """Visualizes the embedding with a title of the model and layer name."""
        unique_labels = list(set(labels))
        markers = {'A': 'o', 'H': 's', 'L': 'D', 'N': '^', 'O': 'v', 'P': '<', 'R': '>'}
        colors = plt.cm.Spectral(np.linspace(0, 1, n_clusters))

        plt.figure(figsize=(12, 10))
        plt.title(f"Embedding Visualization for {model_name} - {layer_name}")

        for letter in unique_labels:
            indices = [i for i, l in enumerate(labels) if l == letter]
            cluster_color = colors[cluster_labels[indices[0]]]
            plt.scatter(embedding[indices, 0], embedding[indices, 1],
                        color=cluster_color,
                        marker=markers.get(letter, 'o'),
                        label=letter,
                        s=80, alpha=1)

            for i in indices:
                plt.text(embedding[i, 0], embedding[i, 1], img_numbers[i], fontsize=8, ha='right', color="black")

        for cluster_idx in range(n_clusters):
            cluster_points = embedding[cluster_labels == cluster_idx]
            if cluster_points.size > 0:
                min_x, min_y = cluster_points.min(axis=0)
                max_x, max_y = cluster_points.max(axis=0)
                padding = 0.1
                plt.gca().add_patch(plt.Rectangle(
                    (min_x - padding, min_y - padding),
                    (max_x - min_x) + 2 * padding,
                    (max_y - min_y) + 2 * padding,
                    fill=False, edgecolor=colors[cluster_idx], linewidth=1.5, linestyle="--"
                ))

        plt.legend(title='Letters', loc='center left', bbox_to_anchor=(1.05, 0.5), fancybox=True, shadow=True, ncol=1)
        plt.grid(True, linestyle='--', alpha=1)
        plt.tight_layout(rect=[0, 0.05, 1, 1])  # Adjust layout to fit the legend
        plt.show()

    def plot_embeddings(self, embeddings, all_clusters, labels, img_numbers, n_clusters, model_to_plot=None, layer_to_plot=None):
        """
        Plots embeddings for each model and layer.
        Optionally filter by specific model or layer.
        """
        for model_name in embeddings:
            if model_to_plot and model_name != model_to_plot:
                continue  # Skip if not the chosen model

            for layer_name, embedding in embeddings[model_name].items():
                if layer_to_plot and layer_name != layer_to_plot:
                    continue  # Skip if not the chosen layer

                cluster_labels = all_clusters[model_name][layer_name]
                self.visualize_embedding(
                    embedding=embedding,
                    labels=labels,
                    cluster_labels=cluster_labels,
                    img_numbers=img_numbers,
                    n_clusters=n_clusters,
                    model_name=model_name,
                    layer_name=layer_name
                )
