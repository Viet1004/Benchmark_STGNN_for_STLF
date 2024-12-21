import matplotlib.pyplot as plt



def plot_time_series(forecast, values, name, title="Time Series Plot", xlabel="Time", ylabel="Value"):
    """
    Plots a time series with a grid.

    Parameters:
    - values (numpy array): The corresponding values for each time step.
    - title (str): The title of the plot.
    - xlabel (str): Label for the x-axis.
    - ylabel (str): Label for the y-axis.
    """

    plt.figure(figsize=(10, 5))
    plt.plot(values, color="blue", linewidth=1.5, label='ground_truth')
    plt.plot(forecast, color="red", linestyle='dashed', linewidth=1.5, label='forecast')
    
    # Title and labels
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    # Adding a grid
    plt.grid(visible=True, which='both', linestyle='--', linewidth=0.5)
    plt.legend()
    # Save plot
    plt.savefig(f'visualization/forecast/{name}.png')

    # Show plot
    plt.show()

def plot_adj_heatmap(adj, file_name, title="Adjacency Matrix Heatmap", cmap="viridis"):
    """
    Plots a heatmap of the adjacency matrix.

    Parameters:
    - adj: numpy.ndarray, adjacency matrix to plot
    - title: str, title of the plot
    - cmap: str, colormap to use for the heatmap
    """
    plt.figure(figsize=(10, 8))
    plt.imshow(adj, aspect='auto', cmap=cmap)
    plt.colorbar(label="Edge Weight")
    plt.title(title)
    plt.xlabel("Node")
    plt.ylabel("Node")
    plt.savefig(file_name)
    print("Print heatmap of adjacency matrix")
    plt.show()
