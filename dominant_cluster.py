#!/usr/bin/env python3
import faiss
from sklearn.cluster import KMeans
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt

import image_utils

def kmeans_faiss(dataset, k, use_gpu):
    """
    Runs KMeans on GPU if available, otherwise on CPU"
    """
    dims = dataset.shape[1]
    cluster = faiss.Clustering(dims, k)
    cluster.verbose = False
    cluster.niter = 20
    cluster.max_points_per_centroid = 10**7

    if use_gpu:
        resources = faiss.StandardGpuResources()
        config = faiss.GpuIndexFlatConfig()
        config.useFloat16 = False
        config.device = 0
        index = faiss.GpuIndexFlatL2(resources, dims, config)
    else:
        index = faiss.IndexFlatL2(dims)

    # perform kmeans
    cluster.train(dataset, index)
    centroids = faiss.vector_float_to_array(cluster.centroids)

    return centroids.reshape(k, dims)

def compute_cluster_assignment(centroids, data, use_gpu):
    dims = centroids.shape[1]

    if use_gpu:
        resources = faiss.StandardGpuResources()
        config = faiss.GpuIndexFlatConfig()
        config.useFloat16 = False
        config.device = 0
        index = faiss.GpuIndexFlatL2(resources, dims, config)
    else:
        index = faiss.IndexFlatL2(dims)

    index.add(centroids)
    _, labels = index.search(data, 1)

    return labels.ravel()

def get_dominant_colors(image, n_clusters=10, use_gpu=False, plot=True):
    # Must pass FP32 data to kmeans_faiss since faiss does not support uint8
    flat_image = image.reshape(
        (image.shape[0] * image.shape[1], 3)).astype(np.float32)

    if use_gpu and faiss.get_num_gpus() > 0:
        centroids = kmeans_faiss(flat_image, n_clusters, use_gpu)
        labels = compute_cluster_assignment(centroids,
                                            flat_image, use_gpu).astype(np.uint8)
        centroids = centroids.astype(np.uint8)
    else:
        clt = KMeans(n_clusters=n_clusters).fit(flat_image)
        centroids = clt.cluster_centers_.astype(np.uint8)
        labels = clt.labels_.astype(np.uint8)

    if plot:
        counts = Counter(labels).most_common()
        centroid_size_tuples = [
            (centroids[k], val / len(labels)) for k, val in counts
        ]
        # This bar_colors function is printing all the extracted colors into a bar plot
        bar_image = image_utils.bar_colors(centroid_size_tuples)
        return centroids, labels, bar_image


    return centroids, labels


def plot_clusters(image, labels, centroids):
    # Reshape labels to the shape of the original image
    labels_reshaped = labels.reshape(image.shape[0], image.shape[1])

    # Calculate number of rows and columns for subplots
    n_clusters = centroids.shape[0]
    n_cols = 5
    n_rows = (n_clusters + n_cols - 1) // n_cols  # Ceiling division

    # Set the figure size to fit 1200 pixels width
    fig_width = 12  # 1200 pixels
    fig_height = 2.4 * n_rows  # Adjust height to maintain aspect ratio
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(fig_width, fig_height))

    # Flatten axes array if there are multiple rows
    if n_rows > 1:
        axes = axes.flatten()

    # Iterate through each cluster index
    for cluster_index in range(n_clusters):
        # Create an image for the current cluster with a white background
        cluster_image = np.ones_like(image) * 255  # Set background to white
        cluster_image[labels_reshaped == cluster_index] = image[labels_reshaped == cluster_index]

        # Display the cluster image
        ax = axes[cluster_index]
        ax.imshow(cluster_image)
        ax.set_title(f'Cluster {cluster_index + 1}', fontsize=14)
        ax.axis('off')

    # Hide any remaining empty subplots
    for ax in axes[n_clusters:]:
        ax.axis('off')

    plt.tight_layout()
    plt.show()