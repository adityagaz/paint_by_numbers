#!/usr/bin/env python3
import numpy as np
import os
import dominant_cluster
import image_utils
import process
from scipy import ndimage as ndi
import matplotlib.pyplot as plt
from colormath.color_objects import sRGBColor, LabColor
from colormath.color_conversions import convert_color
from colormath.color_diff import delta_e_cie2000
from PIL import Image


# Function to convert hex to Lab color
def hex_to_lab(hex_color):
    rgb = sRGBColor.new_from_rgb_hex(hex_color)
    lab = convert_color(rgb, LabColor)
    return lab


# Convert palette to Lab color space
palette_hex = [
    "#907954", "#B59E5F", "#00B89F", "#D2B04C", "#2D2C2F", "#8C83BA", "#DDED1E", "#C65D52",
    "#422D22", "#00675B", "#F7FE00", "#F99471", "#F44741", "#EE9626", "#9BB7D4", "#2A52BE",
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

palette_lab = [hex_to_lab(color) for color in palette_hex]

def rgb_to_lab(pixel):
    rgb = sRGBColor(pixel[0] / 255.0, pixel[1] / 255.0, pixel[2] / 255.0)
    return convert_color(rgb, LabColor)

def lab_to_rgb(lab):
    rgb = convert_color(lab, sRGBColor)
    return (
        int(rgb.clamped_rgb_r * 255),
        int(rgb.clamped_rgb_g * 255),
        int(rgb.clamped_rgb_b * 255)
    )

def find_closest_palette_color(pixel_lab, palette_lab):
    min_delta_e = float('inf')
    closest_color = None
    for color_lab in palette_lab:
        delta_e = delta_e_cie2000(pixel_lab, color_lab)
        if isinstance(delta_e, np.ndarray):
            delta_e = delta_e.item()
        if delta_e < min_delta_e:
            min_delta_e = delta_e
            closest_color = color_lab
    return closest_color

def replace_colors_with_palette(pbn_image, palette_lab):
    rows, cols, _ = pbn_image.shape
    new_image = np.zeros((rows, cols, 3), dtype=np.uint8)

    for i in range(rows):
        for j in range(cols):
            pixel_rgb = pbn_image[i, j]
            pixel_lab = rgb_to_lab(pixel_rgb)
            closest_color_lab = find_closest_palette_color(pixel_lab, palette_lab)
            new_image[i, j] = lab_to_rgb(closest_color_lab)

    return new_image


def simple_matrix_to_image(mat, palette):
    simple_mat_flat = np.array(
        [[col for col in palette[index]] for index in mat.flatten()])
    return simple_mat_flat.reshape(mat.shape + (3,))


def create_cluster_posterize(image_path, clusters=10, pre_blur=True):
    image = image_utils.load_image(image_path, resize=False)
    if pre_blur:
        image = process.blur_image(image)

    dominant_colors, quantized_labels, bar_image = dominant_cluster.get_dominant_colors(
        image, n_clusters=clusters, use_gpu=True, plot=True)

    # Create final PBN image
    plt.imshow(bar_image)
    plt.show()

    print(quantized_labels)

    smooth_labels = process.smoothen(quantized_labels.reshape(image.shape[:-1]))
    smooth_labels_filtered = remove_small_clusters(smooth_labels, threshold_area=50)

    pbn_image = dominant_colors[smooth_labels_filtered].reshape(image.shape)

    plt.imshow(pbn_image)
    plt.show()

    # Replace colors in the PBN image with the closest colors from the palette
    pbn_image_with_palette = replace_colors_with_palette(pbn_image, palette_lab)

    # Creating outline image
    outline_image = outline(pbn_image_with_palette)

    return pbn_image_with_palette, outline_image


def remove_small_clusters(labels, threshold_area):
    # Label connected regions in the smoothed labels
    labels_connected, num_features = ndi.label(labels)

    # Calculate the area of each region
    areas = ndi.sum(labels, labels_connected, range(1, num_features + 1))

    # Filter out regions with area smaller than threshold
    mask = areas >= threshold_area
    labels_filtered = labels.copy()
    labels_filtered[~mask[labels_filtered]] = 0

    return labels_filtered


def are_neighbors_same(mat, x, y):
    width = len(mat[0])
    height = len(mat)
    val = mat[y][x]
    xRel = [1, 0]
    yRel = [0, 1]
    for i in range(0, len(xRel)):
        xx = x + xRel[i]
        yy = y + yRel[i]
        if xx >= 0 and xx < width and yy >= 0 and yy < height:
            if (mat[yy][xx] != val).all():
                return False
    return True


def outline(mat):
    ymax, xmax, _ = mat.shape
    line_mat = np.array([
        255 if are_neighbors_same(mat, x, y) else 0
        for y in range(0, ymax)
        for x in range(0, xmax)
    ],
        dtype=np.uint8)

    return line_mat.reshape((ymax, xmax))


def process_image(input_image_path, output_image_path, num_of_clusters=5, save_outline=False):
    pbn_image, outline_image = create_cluster_posterize(input_image_path, clusters=num_of_clusters)
    image_utils.save_image(pbn_image, output_image_path)

    if save_outline:
        outline_image_path = os.path.splitext(output_image_path)[0] + "_outline.jpg"
        image_utils.save_image(outline_image, outline_image_path)


if __name__ == '__main__':
    input_image_path = f'/Users/adityashandilya/Downloads/testing/test8.jpg'
    output_image_path = f'/Users/adityashandilya/Downloads/testing_output/output_test_1_1_8.jpg'
    num_of_clusters = 5
    save_outline = True
    process_image(input_image_path, output_image_path, num_of_clusters, save_outline)
    print(f"image saved successfully at {output_image_path}")