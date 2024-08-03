import streamlit as st
import numpy as np
import cv2
import process
import image_utils
import dominant_cluster
from PIL import Image
import io
import matplotlib.pyplot as plt
import paint_by_numbers

def simple_matrix_to_image(mat, palette):
    simple_mat_flat = np.array([[col for col in palette[index]] for index in mat.flatten()])
    return simple_mat_flat.reshape(mat.shape + (3,))

def create_cluster_posterize(image, clusters=10, pre_blur=True):
    if pre_blur:
        image = process.blur_image(image)

    dominant_colors, quantized_labels, bar_image = dominant_cluster.get_dominant_colors(
        image, n_clusters=clusters, use_gpu=True, plot=True)

    # Display the color bar
    plt.imshow([dominant_colors])
    plt.show()

    smooth_labels = process.smoothen(quantized_labels.reshape(image.shape[:-1]))
    pbn_image = dominant_colors[smooth_labels].reshape(image.shape)
    outline_image = process_image2(pbn_image, dominant_colors, min_contour_area)  # Generate the outline image

    return pbn_image, outline_image, dominant_colors

def process_image2(pbn_image, dominant_colors, min_contour_area=100):
    gray = cv2.cvtColor(pbn_image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    large_contours = [contour for contour in contours if cv2.contourArea(contour) >= min_contour_area]
    canvas = np.full((gray.shape[0], gray.shape[1], 3), 255, np.uint8)

    for contour in large_contours:
        mask = np.zeros_like(gray)
        cv2.drawContours(mask, [contour], -1, 255, -1)
        mean_color = cv2.mean(pbn_image, mask=mask)[:3]
        closest_color_index = np.argmin(
            [np.linalg.norm(np.array(mean_color) - np.array(color)) for color in dominant_colors])
        bgr_color = tuple(int(c) for c in dominant_colors[closest_color_index])
        cv2.drawContours(canvas, [contour], -1, (0, 0, 0), 1)
        M = cv2.moments(contour)
        if M['m00'] != 0:
            cX = int(M['m10'] / M['m00'])
            cY = int(M['m01'] / M['m00'])
        else:
            cX, cY = 0, 0
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
    ], dtype=np.uint8)
    return line_mat.reshape((ymax, xmax))

st.title("Paint-by-Numbers Image Processor")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

num_clusters = st.slider("Number of Colors", 1, 20, 10)
min_contour_area = st.slider("Minimum Contour Area", 1, 500, 80)

if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    st.image(image, caption='Uploaded Image.', use_column_width=True)
    st.write("")

    if st.button("Process"):
        with st.spinner("Processing..."):
            pbn_image, outline_image, _ = create_cluster_posterize(image, clusters=num_clusters, pre_blur=True)

        st.image(pbn_image, caption='PBN Image.', use_column_width=True)
        st.image(outline_image, caption='Outline Image.', use_column_width=True)

        # Add zoom functionality by displaying the image at a larger size
        st.write("Zoomed PBN Image:")
        st.components.v1.html(
            f"""
            <div style="overflow: auto; width:800px; height:600px">
                <img src="data:image/png;base64,{Image.fromarray(pbn_image).tobytes().decode('utf-8')}" style="width:1000px; height:auto;">
            </div>
            """,
            height=600,
        )

        st.write("Zoomed Outline Image:")
        st.components.v1.html(
            f"""
            <div style="overflow: auto; width:800px; height:600px">
                <img src="data:image/png;base64,{Image.fromarray(outline_image).tobytes().decode('utf-8')}" style="width:1000px; height:auto;">
            </div>
            """,
            height=600,
        )

        pbn_img = Image.fromarray(pbn_image.astype('uint8'))
        buf_pbn = io.BytesIO()
        pbn_img.save(buf_pbn, format="PNG")
        byte_im_pbn = buf_pbn.getvalue()
        st.download_button(label="Download PBN Image", data=byte_im_pbn, file_name="pbn_image.png", mime="image/png")

        outline_img = Image.fromarray(outline_image.astype('uint8'))
        buf_outline = io.BytesIO()
        outline_img.save(buf_outline, format="PNG")
        byte_im_outline = buf_outline.getvalue()
        st.download_button(label="Download Outline Image", data=byte_im_outline, file_name="outline_image.png", mime="image/png")