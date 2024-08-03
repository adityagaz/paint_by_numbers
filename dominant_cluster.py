#!/usr/bin/env python3
import faiss
from sklearn.cluster import KMeans
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import image_utils

import cv2
from skimage import color


def rgb_to_hsv(rgb):
    """
    Convert RGB to HSV color space using OpenCV.
    """
    rgb = np.array(rgb, dtype=np.uint8).reshape(1, 1, 3)
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    return hsv.flatten().astype(float)


def hsv_color_difference(hsv1, hsv2):
    """
    Calculate the perceptual difference between two HSV colors.
    """
    h_diff = min(abs(hsv1[0] - hsv2[0]), 180 - abs(hsv1[0] - hsv2[0])) / 180.0
    s_diff = abs(hsv1[1] - hsv2[1]) / 255.0
    v_diff = abs(hsv1[2] - hsv2[2]) / 255.0
    return h_diff + s_diff + v_diff
def rgb_to_lab(rgb):
    """
    Convert RGB to CIELAB color space using skimage.
    """
    rgb = np.array(rgb, dtype=np.uint8).reshape(1, 1, 3)
    lab = color.rgb2lab(rgb)
    return lab.flatten()
def kmeans_faiss(dataset, k, use_gpu):
    """
    Runs KMeans on GPU if available, otherwise on CPU"
    """
    dims = dataset.shape[1]
    cluster = faiss.Clustering(dims, k)
    cluster.verbose = False
    cluster.niter = 20
    cluster.max_points_per_centroid = 10 ** 7

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


def color_distance(c1, c2):
    """
    Calculate the Euclidean distance between two colors in CIELAB space.
    """
    lab1 = rgb_to_lab(c1)
    lab2 = rgb_to_lab(c2)
    return np.linalg.norm(lab1 - lab2)


def find_closest_palette_color(color, palette, palette_hex):
    """
    Find the closest color in the palette to the given color.
    Print similarity scores for debugging.
    """
    hsv_color = rgb_to_hsv(color)
    distances = [hsv_color_difference(hsv_color, rgb_to_hsv(p)) for p in palette]

    # Debug: Check if distances list is empty
    if not distances:
        print(f"Error: No distances calculated for color {color}")
        return np.array([0, 0, 0])  # Return black or some default color

    # Print the similarity scores
    print(f"Color: {color} - HSV: {hsv_color}")
    for i, dist in enumerate(distances):
        print(f"  Palette Color {palette_hex[i]}: {palette[i]} - Distance: {dist}")

    return palette[np.argmin(distances)]

def replace_colors_with_palette(dominant_colors, palette, palette_hex):
    """
    Replace the dominant colors with the closest colors from the palette.
    """
    replaced_colors = np.array([find_closest_palette_color(c, palette, palette_hex) for c in dominant_colors])
    replaced_colors = np.clip(replaced_colors, 0, 255).astype(np.uint8)
    return replaced_colors


def get_dominant_colors(image, n_clusters=10, use_gpu=False, plot=True):
    if image is None or image.size == 0:
        raise ValueError("Invalid image provided")

    flat_image = image.reshape((image.shape[0] * image.shape[1], 3)).astype(np.float32)

    if use_gpu and faiss.get_num_gpus() > 0:
        centroids = kmeans_faiss(flat_image, n_clusters, use_gpu)
        labels = compute_cluster_assignment(centroids, flat_image, use_gpu).astype(np.uint8)
        centroids = centroids.astype(np.uint8)
    else:
        clt = KMeans(n_clusters=n_clusters).fit(flat_image)
        centroids = clt.cluster_centers_.astype(np.uint8)
        labels = clt.labels_.astype(np.uint8)

    # Define the color palette
    palette_hex = [
        "#FFFDD0", "#F8CEA4", "#F8F6F0", "#F3F4F7", "#FEFFFA", "#F2DEB6", "#C47E5A", "#907954",
        "#B59E5F", "#C28F5A", "#4E3629", "#422D22", "#68553A", "#67492F", "#8D2B00",
        "#907954", "#B59E5F", "#00B89F", "#D2B04C", "#2D2C2F", "#8C83BA", "#DDED1E", "#C65D52",
        "#FBDB32", "#0047AB", "#0047AB", "#478589", "#C47E5A", "#FF4040", "#AE0E36", "#CF3854",
        "#305679", "#5D3954", "#4E3629", "#314F40", "#6A0DAD", "#8B008B", "#009473", "#FFFDD0",
        "#FEDC5A", "#C28F5A", "#CB8E16", "#88B04B", "#858E90", "#036A3E", "#CD5C5C", "#FAB230",
        "#9FAF6C", "#FDF436", "#3B83BD", "#84C3BE", "#D23C77", "#395FA3", "#834655", "#602A7A",
        "#721660", "#F8C81A", "#39FF14", "#FF6700", "#F535AA", "#FFFF00", "#424632", "#FF781F",
        "#898085", "#F8F6F0", "#64B3C9", "#2A894A", "#80C35D", "#EC7F40", "#D33166", "#D97941",
        "#ECAF39", "#468FAA", "#383E99", "#FF0490", "#F8CEA4", "#3D3472", "#F85F4A", "#F1AB38",
        "#68553A", "#C91A09", "#D95030", "#305B3F", "#D84645", "#BCC6CC", "#2271B3", "#4FB9AF",
        "#367588", "#8D2B00", "#F3F4F7", "#3F888F", "#1A2928", "#599FA2", "#2A5198", "#F2DEB6",
        "#67492F", "#E34234", "#E94C3E", "#538F7C", "#330066", "#FFFFFF", "#FFFF00", "#CB9D06",
        "#FEFFFA", "#EDB525", "#C9822A", "#F7D087", "#913832", "#BB3726", "#8A3324", "#D6AD60",
        "#D2C1A3", "#FABEAE", "#FAEBD7"
    ]
    palette_rgb = np.array([mcolors.hex2color(c) for c in palette_hex]) * 255

    # Define skin tone colors
    skin_tone_hex = [
        "#FFFDD0", "#F8CEA4", "#F8F6F0", "#F3F4F7", "#FEFFFA", "#F2DEB6", "#C47E5A", "#907954",
        "#B59E5F", "#C28F5A", "#4E3629", "#422D22", "#68553A", "#67492F", "#8D2B00"
    ]
    skin_tone_rgb = np.array([mcolors.hex2color(c) for c in skin_tone_hex]) * 255

    # Initialize visited arrays
    visited = [False] * len(palette_rgb)
    skin_tone_visited = [False] * len(skin_tone_rgb)

    # Replace the dominant colors with the closest colors from the palette
    replaced_colors = replace_colors_with_palette(centroids, palette_rgb, palette_hex)
    replaced_colors = np.clip(replaced_colors, 0, 255).astype(np.uint8)

    if plot:
        counts = Counter(labels).most_common()
        centroid_size_tuples = [
            (replaced_colors[k], val / len(labels)) for k, val in counts
        ]
        bar_image = image_utils.bar_colors(centroid_size_tuples)

        cluster_areas = [np.sum(labels == i) for i in range(n_clusters)]
        sorted_indices = np.argsort(cluster_areas)[::-1]
        sorted_cluster_areas = sorted(cluster_areas, reverse=True)
        print("Sorted cluster areas (in descending order):", sorted_cluster_areas)

        # Maintain an array of distances
        distances = [find_closest_palette_color_distances(centroid, palette_rgb) for centroid in centroids]

        # Color mapping with special handling for skin tones
        color_mapping = {}
        for cluster_index in sorted_indices:
            sorted_distances = np.argsort(distances[cluster_index])
            is_skin_tone = any(
                np.allclose(centroids[cluster_index], skin_color, atol=15) for skin_color in skin_tone_rgb)

            if is_skin_tone:
                for palette_index, skin_color in enumerate(skin_tone_rgb):
                    if not skin_tone_visited[palette_index]:
                        color_mapping[cluster_index] = skin_color
                        skin_tone_visited[palette_index] = True
                        break
                else:
                    # If all skin tones are visited, pick the closest available skin tone
                    for skin_color, visited_flag in zip(skin_tone_rgb, skin_tone_visited):
                        if not visited_flag:
                            color_mapping[cluster_index] = skin_color
                            break
            else:
                for palette_index in sorted_distances:
                    if not visited[palette_index]:
                        color_mapping[cluster_index] = palette_rgb[palette_index]
                        visited[palette_index] = True
                        break

        print("Color mapping (cluster_index -> palette_color):")
        for cluster_index, palette_color in color_mapping.items():
            print(f"Cluster {cluster_index} -> {palette_color}")

        labels_reshaped = labels.reshape(image.shape[0], image.shape[1])
        new_image = np.zeros_like(image)
        for cluster_index, palette_color in color_mapping.items():
            new_image[labels_reshaped == cluster_index] = palette_color

        return replaced_colors, labels, bar_image
    return replaced_colors, labels


def find_closest_palette_color_distances(centroid, palette):
    hsv_centroid = rgb_to_hsv(centroid)
    return [hsv_color_difference(hsv_centroid, rgb_to_hsv(p)) for p in palette]


def find_closest_palette_color_distances(centroid, palette):
    hsv_centroid = rgb_to_hsv(centroid)
    return [hsv_color_difference(hsv_centroid, rgb_to_hsv(p)) for p in palette]


def find_closest_palette_color_distances(centroid, palette):
    hsv_centroid = rgb_to_hsv(centroid)
    return [hsv_color_difference(hsv_centroid, rgb_to_hsv(p)) for p in palette]

def find_closest_palette_color_distances(centroid, palette):
    hsv_centroid = rgb_to_hsv(centroid)
    return [hsv_color_difference(hsv_centroid, rgb_to_hsv(p)) for p in palette]


def find_closest_palette_color(color, palette, palette_hex):
    """
    Find the closest color in the palette to the given color.
    Print similarity scores for debugging.
    """
    hsv_color = rgb_to_hsv(color)
    distances = [hsv_color_difference(hsv_color, rgb_to_hsv(p)) for p in palette]

    # Debug: Check if distances list is empty
    if not distances:
        print(f"Error: No distances calculated for color {color}")
        return np.array([0, 0, 0])  # Return black or some default color

    # Print the similarity scores
    print(f"Color: {color} - HSV: {hsv_color}")
    for i, dist in enumerate(distances):
        print(f"  Palette Color {palette_hex[i]}: {palette[i]} - Distance: {dist}")

    return palette[np.argmin(distances)]

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
