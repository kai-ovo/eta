import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import numpy as np


def plot_setting(usetex=False):
    """
    Plotting settings
    """
    mpl.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['DejaVu Serif'],
    'mathtext.fontset': 'cm',  # Computer Modern font for math
    'text.usetex': usetex,  # Disable LaTeX rendering
    'mathtext.default': 'it'  # Regular math text
})

    # Set Seaborn's theme: white background, no grid, and gray edges.
    sns.set_theme(style="white", rc={
        "axes.grid": False,        # Remove grid lines
        "axes.facecolor": "white", # White background for the axes
        "axes.edgecolor": "gray"   # Gray edges around the plot
    })

    # mpl.rcParams['axes.labelsize'] = 12
    # mpl.rcParams['font.size'] = 12
    # mpl.rcParams['axes.titlesize'] = 14
    # mpl.rcParams['xtick.labelsize'] = 10
    # mpl.rcParams['ytick.labelsize'] = 10
    # mpl.rcParams['legend.fontsize'] = 10
    # mpl.rcParams['figure.titlesize'] = 16
    # mpl.rcParams['figure.dpi'] = 400  # Higher resolution for publication quality


def plot_pdf(y, py, linestyle=None, color=None,
             label=None, labelsize=16, 
             title=None, titlesize=30, 
             fig=None, ax=None, figsize=(12,5)):
    """ 
    inputs:
        y - ndarray, provided by get_output_pdf
        py - pdf value of array y
        label - if provided, figure will draw legend for the pdf curve
        title - suptitle
        fig, ax - if provided, will continue drawing on fig
    """
    if fig is None:
        fig, ax = plt.subplots(1,2,figsize=figsize,dpi=300)
    
    if fig is not None:
        assert ax is not None 
    if ax is not None:
        assert fig is not None

    if linestyle is not None:
        ax[0].plot(y, py, color=color, label=label, linestyle=linestyle)#, color='royalblue')
        ax[1].plot(y, py, color=color, linestyle=linestyle)#, color='royalblue')
    else:
        ax[0].plot(y, py, color=color, label=label)#, color='royalblue')
        ax[1].plot(y, py, color=color)#, color='royalblue')
        
    ax[0].set_title('Linear Scale', fontsize=24)
    ax[0].set_xlabel(r"$y$", fontsize=24)

    ax[1].set_yscale('log')
    ax[1].set_title('Log Scale', fontsize=24)
    ax[1].set_xlabel(r"$y$", fontsize=24)
    if title is not None:
        fig.suptitle(title, fontsize=titlesize)
    if label is not None:
        ax[0].legend(prop={'size':labelsize})

    return fig, ax

def plot_2contours(x1, x2, y1, y2, 
                   x_scatter = None,
                   title=None, titlesize=30, 
                   figsize=(11,5.5), ax_lim = (-6,6), nlines=48):
    """ 
    inputs:
        x1, x2: meshgrid indexed by 'ij' 
        y1 - first (true) field value
        y2 - second (comparing) field value
        title - suptitle
        fig, ax - if provided, will continue drawing on fig
    """
    fig, ax = plt.subplots(1,2,figsize=figsize, dpi=300,
                            gridspec_kw={'width_ratios': [4.3, 5]})

    N = x1.shape[0]
    plt1 = ax[0].contour(x1.numpy(), x2.numpy(), y1.reshape(N,N).numpy(), nlines, cmap='Reds')
    cb1 = fig.colorbar(plt1)
    ax[0].set_xlim(ax_lim)
    ax[0].set_ylim(ax_lim)
    ax[0].grid(linestyle='--', linewidth=0.5)

    if x_scatter is not None: # mark training data
        n_scatter = x_scatter.shape[0]
        ax[0].plot(x_scatter[:,0], x_scatter[:,1], 'x', markersize=4, color='blue', alpha=1, 
                   label=r'Training Data ($n$' + f' = {n_scatter})')
        ax[0].legend(prop={'size':16})

    plt2 = ax[1].contour(x1.numpy(),x2.numpy(), y2.detach().cpu().reshape(N,N).numpy(), nlines, cmap='Reds')
    cb2 = fig.colorbar(plt2)
    cb2.mappable.set_clim(*cb1.mappable.get_clim()) # set on the same scale
    cb1.remove() # remove the first color bar
    ax[1].set_xlim(ax_lim)
    ax[1].set_ylim(ax_lim)
    ax[1].grid(linestyle='--', linewidth=0.5)

    if title is not None:
        fig.suptitle(title, fontsize=titlesize)

    return fig, ax

def plot_e2a_contour(x1, x2, output_on_grid,
                     figsize=(6.5,5.5), ax_lim = (-8,8), nlines=48,
                     title=None, titlesize=24):
    """
        plot E2A-NN prediction only
        
    """
    fig, ax = plt.subplots(1,1,figsize=figsize, dpi=300)
    plt2 = ax.contour(x1.numpy(),x2.numpy(), output_on_grid.detach().cpu().numpy(), nlines, cmap='Reds')
    fig.colorbar(plt2)
    ax.set_xlim(ax_lim)
    ax.set_ylim(ax_lim)
    ax.grid(linestyle='--', linewidth=0.5)

    if title is not None:
        fig.suptitle(title, fontsize=titlesize)
    fig.tight_layout()
    return fig, ax

def add_true_vs_e2a_title(fig, ax,  tight_layout=True):
    ax[0].set_title("True", fontsize=24)
    ax[1].set_title("Test", fontsize=24)
#     fig.suptitle("E2A-NN Prediction", fontsize=30)
    if tight_layout:
        fig.tight_layout()
    return fig, ax    

def add_pretrain_title(fig, ax):
    ax[0].set_title("True Random Field", fontsize=24)
    ax[1].set_title("Pretrained E2A-NN Output", fontsize=24)
    fig.suptitle("Pretraining E2A-NN with Gaussian Random Field", fontsize=30)
    fig.tight_layout()

    return fig, ax

def plot_nfields(*snapshots, title_date=None, latitudes=None, longitudes=None, cmap="jet", title=None, add_ylabel=False):
    if latitudes is None or longitudes is None:
        raise ValueError("Latitudes and longitudes must be provided.")
    
    if len(snapshots) == 0:
        raise ValueError("At least one snapshot must be provided.")

    # Determine global min and max from all snapshots for consistent coloring
    combined_min = np.min([s.min() for s in snapshots])
    combined_max = np.max([s.max() for s in snapshots])
    if combined_min == combined_max:
        combined_max += 1e-3

    n_fields = len(snapshots)
    fig, axes = plt.subplots(
        nrows=n_fields, ncols=1, figsize=(10, 4 * n_fields),
        subplot_kw={'projection': ccrs.PlateCarree()},
        constrained_layout=True,
        dpi=400
    )
    if n_fields == 1:
        axes = [axes]

    for ax, snapshot in zip(axes, snapshots):
        ax.add_feature(cfeature.STATES, edgecolor='#d3d3d3', linewidth=1)
        ax.add_feature(cfeature.COASTLINE, edgecolor='#d3d3d3', linewidth=1)

        nlat, nlon = snapshot.shape

        # If snapshot matches the full domain (80x160), just plot directly
        if nlat == latitudes.size and nlon == longitudes.size:
            lat_plot = latitudes
            lon_plot = longitudes
        else:
            # Otherwise, re-map it so that the 8x16 covers the full bounding box
            lat_plot = np.linspace(latitudes.max(), latitudes.min(), nlat)
            lon_plot = np.linspace(longitudes.min(), longitudes.max(), nlon)


        img = ax.pcolormesh(lon_plot, lat_plot, snapshot,
                            cmap=cmap,
                            vmin=combined_min, vmax=combined_max,
                            transform=ccrs.PlateCarree(),
                            shading='auto')

        ax.set_extent([lon_plot.min(), lon_plot.max(),
                       lat_plot.min(), lat_plot.max()],
                      crs=ccrs.PlateCarree())
    
    if title_date:
        axes[0].set_title(f"Peak Precipitation (mm) on {title_date}", fontsize=24)

    if title is not None:
        fig.suptitle(title, fontsize=24)

    if add_ylabel:
        labels = [
            "Coarse Input",
            "Ground Truth",
            "MSE-Map Output",
            r"$\eta$-Map Output",
        ]

        for ax, lbl in zip(axes, labels):
            ax.text(-0.05, 0.5, lbl, transform=ax.transAxes,
                    fontsize=26, va='center', ha='right', rotation=90)

    plt.tight_layout()
    cbar = plt.colorbar(img, ax=axes, orientation='vertical',
                        fraction=0.04 if n_fields == 1 else 0.01 * n_fields, pad=0.02)
    tick_pos = np.linspace(combined_min, combined_max, 6)
    cbar.set_ticks(tick_pos)
    cbar.ax.tick_params(labelsize=14)
    return fig, axes

def plot_contour(data_snapshot, title_date=None, latitudes=None, longitudes=None):
    """
    Function to create a contour plot for a single snapshot of precipitation data.
    
    Parameters:
    data_snapshot (numpy array): 2D numpy array of precipitation data at a specific time.
    latitudes (numpy array): 1D array of latitude values.
    longitudes (numpy array): 1D array of longitude values.
    title_date (str): Date string for the title of the plot, e.g., '1999-06-15'.
    """
    # Set up the map for plotting
    fig, ax = plt.subplots(figsize=(10, 6), subplot_kw={'projection': ccrs.PlateCarree()}, dpi=300)

    # Add features to the map (state borders, coastlines)
    ax.add_feature(cfeature.STATES, edgecolor='black', linewidth=1)
    ax.add_feature(cfeature.COASTLINE, edgecolor='black', linewidth=1)

    # Plot the precipitation data
    contour = ax.contourf(longitudes, latitudes, data_snapshot, cmap='Blues', transform=ccrs.PlateCarree())

    # Adjust color bar size and position
    cbar = plt.colorbar(contour, ax=ax, orientation='vertical', fraction=0.02, pad=0.04)
    
    # Set the extent to match the spatial range
    ax.set_extent([longitudes.min(), longitudes.max(), latitudes.min(), latitudes.max()], crs=ccrs.PlateCarree())

    # Add title (date without hour information)
    if title_date:
        plt.title(f"Peak Precipitation (mm) on {title_date}", fontsize=16)
    else:
        plt.title(f"Peak Precipitation (mm)", fontsize=16)

    # Apply tight layout
    plt.tight_layout()

    # Show the plot
    plt.show()
