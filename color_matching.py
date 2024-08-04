import matplotlib.pyplot as plt
import numpy as np
import colorsys


def hex_to_rgb(hex_color):
    hex_color = hex_color.lstrip('#')
    return tuple(int(hex_color[i:i + 2], 16) for i in (0, 2, 4))


def rgb_to_hex(rgb_color):
    return '#{:02x}{:02x}{:02x}'.format(*rgb_color)


def rgb_to_hsv(r, g, b):
    return colorsys.rgb_to_hsv(r / 255.0, g / 255.0, b / 255.0)


def color_difference(color1, color2):
    return np.sqrt(sum((e1 - e2) ** 2 for e1, e2 in zip(color1, color2)))


def find_closest_color(input_color, palette):
    input_rgb = hex_to_rgb(input_color)
    input_hsv = rgb_to_hsv(*input_rgb)

    differences = []

    for color in palette:
        palette_rgb = hex_to_rgb(color)
        palette_hsv = rgb_to_hsv(*palette_rgb)

        rgb_diff = color_difference(input_rgb, palette_rgb)
        hsv_diff = color_difference(input_hsv, palette_hsv)

        avg_diff = (rgb_diff + hsv_diff) / 2
        differences.append((avg_diff, color))

    differences.sort()
    closest_color = differences[0][1]
    return closest_color


def plot_color_mappings(input_colors, mapped_colors):
    num_colors = len(input_colors)
    fig, axes = plt.subplots(30, 2, figsize=(20, 5))

    for i in range(num_colors):
        input_rgb = np.array([[hex_to_rgb(input_colors[i])]], dtype=np.uint8)
        closest_rgb = np.array([[hex_to_rgb(mapped_colors[i])]], dtype=np.uint8)

        axes[i, 0].imshow(input_rgb, aspect='auto')
        axes[i, 0].set_title(f'Original Color: {input_colors[i]}')
        axes[i, 0].axis('off')

        axes[i, 1].imshow(closest_rgb, aspect='auto')
        axes[i, 1].set_title(f'Mapped Color: {mapped_colors[i]}')
        axes[i, 1].axis('off')

    plt.tight_layout()
    plt.show()


# Example usage
test_rgb_colors = [
    [235, 237, 229],
    [51, 47, 44],
    [214, 163, 120],
    [238, 185, 143],
    [184, 130, 92],
    [21, 22, 23],
    [187, 197, 176],
    [134, 115, 109],
    [248, 176, 84],
    [146, 96, 60],
    [245, 117, 34],
    [162, 154, 145],
    [198, 226, 211],
    [89, 77, 69],
    [246, 215, 165]
]
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
# Convert test RGB colors to hex
test_hex_colors = [rgb_to_hex(color) for color in test_rgb_colors]

# Find closest colors
mapped_hex_colors = [find_closest_color(color, palette_hex) for color in test_hex_colors]

# Plot original vs mapped colors
plot_color_mappings(test_hex_colors, mapped_hex_colors)