import matplotlib.pyplot as plt
import numpy as np

def visualize_colors(colors):
    """
    Visualize a list of RGB colors.

    Args:
        colors (list): A list of RGB colors, where each color is a list of three integers in the range [0, 255].

    Returns:
        None
    """
    # Convert the colors to a numpy array
    colors = np.array(colors)

    # Create a figure with a single axis
    fig, ax = plt.subplots()

    # Create a color palette with the given colors
    palette = plt.cm.get_cmap(None, len(colors))

    # Set the colors of the palette
    palette.colors = colors / 255  # Normalize the colors to the range [0, 1]

    # Create a color bar with the palette
    cbar = ax.imshow(np.arange(len(colors)).reshape(1, -1), cmap=palette, aspect='auto')

    # Remove the axis ticks
    ax.set_xticks([])
    ax.set_yticks([])

    # Show the plot
    plt.show()

# Example usage:
colors = [
    [31, 31, 31],
    [241, 241, 241],
    [116, 116, 116],
    [179, 179, 179],
    [60, 60, 60]
]

visualize_colors(colors)