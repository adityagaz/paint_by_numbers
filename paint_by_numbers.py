#!/usr/bin/env python3
import os
import dominant_cluster
import image_utils
import process
from scipy import ndimage as ndi
import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage
import cv2

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

    # Display the color bar
    print(type(dominant_colors))
    plt.imshow([dominant_colors])
    plt.show()

    # Update dominant colors
    # dominant_colors = [
    #     [250, 235, 215], [104, 85, 58], [248, 206, 164], [26, 41, 40], [194, 143, 90]
    # ]
    # dominant_colors = np.array(dominant_colors, dtype=np.uint8)
    print(dominant_colors)
    #printing clusters one by one
    dominant_cluster.plot_clusters(image, quantized_labels, dominant_colors)

    smooth_labels = process.smoothen(quantized_labels.reshape(image.shape[:-1]))
    pbn_image = dominant_colors[smooth_labels].reshape(image.shape)

    # Convert pbn_image to uint8
    pbn_image = pbn_image.astype(np.uint8)

    return pbn_image, dominant_colors

def process_image2(pbn_image, dominant_colors, min_contour_area=100):
    # Convert the image to grayscale
    gray = cv2.cvtColor(pbn_image, cv2.COLOR_BGR2GRAY)

    # Edge detection using Canny
    edges = cv2.Canny(gray, 50, 150)

    # Find contours
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Filter out small contours based on area
    large_contours = [contour for contour in contours if cv2.contourArea(contour) >= min_contour_area]

    # Prepare a canvas for drawing contours
    canvas = np.full((gray.shape[0], gray.shape[1], 3), 255, np.uint8)  # White background

    for contour in large_contours:
        mask = np.zeros_like(gray)
        cv2.drawContours(mask, [contour], -1, 255, -1)
        mean_color = cv2.mean(pbn_image, mask=mask)[:3]
        closest_color_index = np.argmin(
            [np.linalg.norm(np.array(mean_color) - np.array(color)) for color in dominant_colors])

        # Ensure color is in the right format (BGR integer tuple)
        bgr_color = tuple(int(c) for c in dominant_colors[closest_color_index])

        # Draw the contour
        cv2.drawContours(canvas, [contour], -1, (0, 0, 0), 1)  # Draw all contours in black to avoid color issues

        # Calculate centroid and label the contour
        M = cv2.moments(contour)
        if M['m00'] != 0:
            cX = int(M['m10'] / M['m00'])
            cY = int(M['m01'] / M['m00'])
        else:
            cX, cY = 0, 0  # Fallback if contour is very small
        cv2.putText(canvas, str(closest_color_index), (cX, cY), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

    return canvas

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


def process_image(input_image_path, output_image_path, num_of_clusters=5, save_outline=False, min_contour_area=80):
    pbn_image, dominant_colors = create_cluster_posterize(input_image_path, clusters=num_of_clusters)
    image_utils.save_image(pbn_image, output_image_path)

    if save_outline:
        outline_image = process_image2(pbn_image, dominant_colors, min_contour_area)
        outline_image_path = os.path.splitext(output_image_path)[0] + "_outline.jpg"
        image_utils.save_image(outline_image, outline_image_path)


input_image_path = '/Users/adityashandilya/Downloads/testing/test9.jpg'
output_image_path = '/Users/adityashandilya/Downloads/testing_output/output_test_1_1_9.jpg'
num_of_clusters = 15
save_outline = True
min_contour_area = 80

 # Adjust this value based on the desired minimum contour size
process_image(input_image_path, output_image_path, num_of_clusters, save_outline, min_contour_area)
print(f"Image saved successfully at {output_image_path}")
if save_outline:
    print(f"Outline image also saved successfully.")