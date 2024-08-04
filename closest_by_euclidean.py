import numpy as np


# Convert hex to RGB
def hex_to_rgb(hex_code):
    hex_code = hex_code.lstrip('#')
    return tuple(int(hex_code[i:i + 2], 16) for i in (0, 2, 4))


# Calculate Euclidean distance
def euclidean_distance(c1, c2):
    return np.sqrt(sum((a - b) ** 2 for a, b in zip(c1, c2)))


# Find the closest color
def find_closest_colors(rgb_values, palette):
    palette_rgb = [hex_to_rgb(color) for color in palette]
    closest_colors = []
    for rgb in rgb_values:
        distances = [euclidean_distance(rgb, palette_color) for palette_color in palette_rgb]
        closest_color_index = np.argmin(distances)
        closest_colors.append(palette[closest_color_index])
    return closest_colors


palette = [
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

rgb_values = [
    [227, 230, 213],
    [106, 80, 64],
    [234, 192, 149],
    [24, 22, 21],
    [180, 136, 105]
]

closest_colors = find_closest_colors(rgb_values, palette)
print("Closest colors in the palette are:", closest_colors)
