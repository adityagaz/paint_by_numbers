import numpy as np
import os
import dominant_cluster
import image_utils
import process
from scipy import ndimage as ndi
import matplotlib.pyplot as plt
from colormath.color_objects import LabColor, sRGBColor
from colormath.color_conversions import convert_color
from skimage.color import rgb2lab, lab2rgb
from matplotlib.colors import to_rgb

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

def hex_to_lab(hex_color):
    rgb = to_rgb(hex_color)
    rgb = np.array(rgb).reshape(1, 1, 3)
    lab = rgb2lab(rgb)
    return lab[0, 0]

def delta_e_cie2000(lab1, lab2):
    lab1_array = np.array(lab1)
    lab2_array = np.array(lab2)
    return np.linalg.norm(lab1_array - lab2_array)

def convert_to_lab(hex_color):
    """
    Convert a hex color to CIELAB color space.
    """
    rgb = sRGBColor.new_from_rgb_hex(hex_color)
    lab = convert_color(rgb, LabColor)
    return lab

def compute_distance(lab1, lab2):
    """
    Compute the Euclidean distance between two CIELAB colors.
    """
    return np.sqrt((lab1.lab_l - lab2.lab_l) ** 2 + (lab1.lab_a - lab2.lab_a) ** 2 + (lab1.lab_b - lab2.lab_b) ** 2)

def find_closest_palette_color(pixel_lab, palette_lab):
    """
    Find the closest palette color to a given pixel color using the CIELAB color similarity algorithm.
    """
    min_distance = float('inf')
    closest_color = None
    for color_lab in palette_lab:
        distance = compute_distance(pixel_lab, color_lab)
        if distance < min_distance:
            min_distance = distance
            closest_color = color_lab
    return closest_color

def replace_colors_with_palette(pbn_image, palette_hex):

    """
    Replace the colors in the PBN image with the closest colors from the palette using the CIELAB color similarity algorithm.
    """
    palette_lab = [convert_to_lab(color) for color in palette_hex]
    pbn_image_with_palette = np.zeros_like(pbn_image)
    for i in range(pbn_image.shape[0]):
        for j in range(pbn_image.shape[1]):
            pixel_rgb = pbn_image[i, j] / 255.0
            pixel_lab = convert_to_lab(rgb2hex(pixel_rgb))
            closest_color_lab = find_closest_palette_color(pixel_lab, palette_lab)
            closest_color_rgb = lab2rgb(closest_color_lab)
            pbn_image_with_palette[i, j] = (np.array(closest_color_rgb) * 255).astype(np.uint8)
    return pbn_image_with_palette

def rgb2hex(rgb):
    return '#{:02x}{:02x}{:02x}'.format(int(rgb[0] * 255), int(rgb[1] * 255), int(rgb[2] * 255))

def lab2rgb(lab):
    rgb = convert_color(lab, sRGBColor)
    return [rgb.clamped_rgb_r, rgb.clamped_rgb_g, rgb.clamped_rgb_b]

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

    smooth_labels = process.smoothen(quantized_labels.reshape(image.shape[:-1]))
    smooth_labels_filtered = remove_small_clusters(smooth_labels, threshold_area=50)

    pbn_image = dominant_colors[smooth_labels_filtered].reshape(image.shape)
    mapped_image = replace_colors_with_palette(pbn_image , palette_hex)
    plt.imshow(pbn_image)
    plt.show()

    return mapped_image, outline(mapped_image)

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
    input_image_path = f'/Users/adityashandilya/Downloads/testing/test9.jpg'
    output_image_path = f'/Users/adityashandilya/Downloads/testing_output/output_test_1_1_9.jpg'
    num_of_clusters = 5
    save_outline = True
    process_image(input_image_path, output_image_path, num_of_clusters, save_outline)
    print(f"image saved successfully at {output_image_path}")