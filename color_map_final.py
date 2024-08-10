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


def find_closest_color(input_color, palette):
    input_rgb = hex_to_rgb(input_color)
    input_xyz = rgb_to_xyz(*input_rgb)
    input_lab = xyz_to_lab(*input_xyz)

    differences = []

    for color in palette:
        palette_rgb = hex_to_rgb(color)
        palette_xyz = rgb_to_xyz(*palette_rgb)
        palette_lab = xyz_to_lab(*palette_xyz)

        diff = delta_e94(input_lab, palette_lab)
        differences.append((diff, color))

    differences.sort()
    closest_color = differences[0][1]
    closest_distance = differences[0][0]
    return closest_color, closest_distance


def replace_colors_with_palette(centroids, palette_rgb, palette_hex):
    replaced_colors = []
    distances = []

    for centroid in centroids:
        centroid_hex = rgb_to_hex(centroid)
        closest_color_hex, closest_distance = find_closest_color(centroid_hex, palette_hex)
        closest_color_rgb = hex_to_rgb(closest_color_hex)
        replaced_colors.append(closest_color_rgb)
        distances.append(closest_distance)

    return np.array(replaced_colors), np.array(distances)