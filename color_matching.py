import matplotlib.pyplot as plt
import numpy as np

# Constants for the reference white point (D65 illuminant)
REF_X = 95.047
REF_Y = 100.000
REF_Z = 108.883


def hex_to_rgb(hex_color):
    hex_color = hex_color.lstrip('#')
    return tuple(int(hex_color[i:i + 2], 16) for i in (0, 2, 4))


def rgb_to_hex(rgb_color):
    return '#{:02x}{:02x}{:02x}'.format(*rgb_color)


def rgb_to_xyz(r, g, b):
    var_R = r / 255.0
    var_G = g / 255.0
    var_B = b / 255.0

    if var_R > 0.04045:
        var_R = ((var_R + 0.055) / 1.055) ** 2.4
    else:
        var_R = var_R / 12.92
    if var_G > 0.04045:
        var_G = ((var_G + 0.055) / 1.055) ** 2.4
    else:
        var_G = var_G / 12.92
    if var_B > 0.04045:
        var_B = ((var_B + 0.055) / 1.055) ** 2.4
    else:
        var_B = var_B / 12.92

    var_R = var_R * 100
    var_G = var_G * 100
    var_B = var_B * 100

    X = var_R * 0.4124 + var_G * 0.3576 + var_B * 0.1805
    Y = var_R * 0.2126 + var_G * 0.7152 + var_B * 0.0722
    Z = var_R * 0.0193 + var_G * 0.1192 + var_B * 0.9505

    return X, Y, Z


def xyz_to_lab(X, Y, Z):
    var_X = X / REF_X
    var_Y = Y / REF_Y
    var_Z = Z / REF_Z

    if var_X > 0.008856:
        var_X = var_X ** (1 / 3)
    else:
        var_X = (7.787 * var_X) + (16 / 116)
    if var_Y > 0.008856:
        var_Y = var_Y ** (1 / 3)
    else:
        var_Y = (7.787 * var_Y) + (16 / 116)
    if var_Z > 0.008856:
        var_Z = var_Z ** (1 / 3)
    else:
        var_Z = (7.787 * var_Z) + (16 / 116)

    L = (116 * var_Y) - 16
    a = 500 * (var_X - var_Y)
    b = 200 * (var_Y - var_Z)

    return L, a, b


def delta_e94(lab1, lab2, WHT_L=1, WHT_C=1, WHT_H=1):
    L1, a1, b1 = lab1
    L2, a2, b2 = lab2

    xC1 = np.sqrt(a1 ** 2 + b1 ** 2)
    xC2 = np.sqrt(a2 ** 2 + b2 ** 2)
    xDL = L2 - L1
    xDC = xC2 - xC1
    xDE = np.sqrt((L1 - L2) ** 2 + (a1 - a2) ** 2 + (b1 - b2) ** 2)
    xDH = xDE ** 2 - xDL ** 2 - xDC ** 2

    if xDH > 0:
        xDH = np.sqrt(xDH)
    else:
        xDH = 0

    xSC = 1 + 0.045 * xC1
    xSH = 1 + 0.015 * xC1
    xDL /= WHT_L
    xDC /= (WHT_C * xSC)
    xDH /= (WHT_H * xSH)

    delta_E94 = np.sqrt(xDL ** 2 + xDC ** 2 + xDH ** 2)

    return delta_E94


def find_closest_color(input_colors, palette):
    closest_colors = []
    closest_indices = []
    for input_color in input_colors:
        input_hex = rgb_to_hex(input_color)  # Convert RGB to hex
        input_rgb = hex_to_rgb(input_hex)
        input_xyz = rgb_to_xyz(*input_rgb)
        input_lab = xyz_to_lab(*input_xyz)
        differences = []
        for idx, color in enumerate(palette):
            palette_rgb = hex_to_rgb(color)
            palette_xyz = rgb_to_xyz(*palette_rgb)
            palette_lab = xyz_to_lab(*palette_xyz)
            diff = delta_e94(input_lab, palette_lab)
            differences.append((diff, color, idx + 1))
        differences.sort()
        closest_color = differences[0][1]
        closest_index = differences[0][2]
        closest_colors.append(closest_color)
        closest_indices.append(closest_index)
    return closest_colors, closest_indices


def plot_color_mappings(input_colors, mapped_colors, mapped_indices):
    num_colors = len(input_colors)
    fig, axes = plt.subplots(2, num_colors, figsize=(20, 5))

    for i in range(num_colors):
        input_rgb = np.array([[hex_to_rgb(input_colors[i])]], dtype=np.uint8)
        closest_rgb = np.array([[hex_to_rgb(mapped_colors[i])]], dtype=np.uint8)

        axes[0, i].imshow(input_rgb, aspect='auto')
        axes[0, i].axis('off')

        axes[1, i].imshow(closest_rgb, aspect='auto')
        axes[1, i].axis('off')

    plt.tight_layout()
    plt.show()


# Comprehensive test case covering the full spectrum of colors
test_rgb_colors_full_spectrum = [
    (254, 202, 182),
    (245, 185, 158),
    (246, 171, 142),
    (217, 118, 76),
    (252, 188, 140),
    (229, 144, 90),
    (228, 131, 86),
    (178, 85, 44),
    (143, 70, 29),
    (132, 57, 17),
    (245, 194, 151),
    (234, 154, 95),
    (208, 111, 56),
    (168, 66, 17),
    (147, 68, 27),
    (250, 220, 196),
    (245, 187, 149),
    (239, 165, 128),
    (203, 137, 103),
    (168, 102, 68),
    (247, 188, 154),
    (243, 160, 120),
    (213, 144, 75),
    (154, 79, 48),
    (127, 67, 41),
    (255, 218, 179),
    (250, 187, 134),
    (244, 159, 104),
    (198, 110, 46),
    (138, 67, 3),
    (253, 196, 179),
    (245, 158, 113),
    (236, 134, 86),
    (182, 88, 34),
    (143, 60, 18),
    (236, 162, 113),
    (233, 132, 86),
    (219, 116, 75),
    (205, 110, 66),
    (173, 83, 46),
    (242, 207, 175),
    (215, 159, 102),
    (208, 138, 86),
    (195, 134, 80),
    (168, 112, 63),
    (247, 168, 137),
    (221, 132, 98),
    (183, 90, 57),
    (165, 87, 51),
    (105, 29, 15),
]

# Convert test RGB colors to hex
test_hex_colors_full_spectrum = [rgb_to_hex(color) for color in test_rgb_colors_full_spectrum]

# Define the palette (using previously defined palette_hex)
palette_hex = [
    "#907954", "#B59E5F", "#00B89F", "#D2B04C", "#2D2C2F", "#8C83BA", "#DDED1E", "#C65D52", "#422D22", "#00675B",
    "#F7FE00", "#F99471", "#F44741", "#EE9626", "#9BB7D4", "#2A52BE", "#FBDB32", "#0047AB", "#0047AB", "#478589",
    "#C47E5A", "#FF4040", "#AE0E36", "#CF3854", "#305679", "#5D3954", "#4E3629", "#314F40", "#6A0DAD", "#8B008B",
    "#009473", "#FFFDD0", "#FEDC5A", "#C28F5A", "#CB8E16", "#88B04B", "#858E90", "#036A3E", "#CD5C5C", "#FAB230",
    "#9FAF6C", "#FDF436", "#3B83BD", "#84C3BE", "#D23C77", "#395FA3", "#834655", "#602A7A", "#721660", "#F8C81A",
    "#39FF14", "#FF6700", "#F535AA", "#FFFF00", "#424632", "#FF781F", "#898085", "#F8F6F0", "#64B3C9", "#2A894A",
    "#80C35D", "#EC7F40", "#D33166", "#D97941", "#ECAF39", "#468FAA", "#383E99", "#FF0490", "#F8CEA4", "#3D3472",
    "#F85F4A", "#F1AB38", "#68553A", "#C91A09", "#D95030", "#305B3F", "#D84645", "#BCC6CC", "#2271B3", "#4FB9AF",
    "#367588", "#8D2B00", "#F3F4F7", "#3F888F", "#1A2928", "#599FA2", "#2A5198", "#F2DEB6", "#67492F", "#E34234",
    "#E94C3E", "#538F7C", "#330066", "#FFFFFF", "#FFFF00", "#CB9D06", "#FEFFFA", "#EDB525", "#C9822A", "#F7D087",
    "#913832", "#BB3726", "#8A3324", "#D6AD60", "#D2C1A3", "#FABEAE", "#FAEBD7", "#582617"
]

# Find closest colors and their indices
mapped_colors, mapped_indices = find_closest_color(test_rgb_colors_full_spectrum, palette_hex)
# Plot original vs mapped colors with indices
plot_color_mappings(test_hex_colors_full_spectrum, mapped_colors, mapped_indices)
