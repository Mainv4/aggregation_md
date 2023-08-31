import matplotlib.pyplot as plt

def set_plot_style():
    """
    Set the plot style
    """
    plt.style.use('ggplot')
    plt.rcParams['figure.dpi'] = 200
    plt.rcParams['lines.linewidth'] = 2
    plt.rcParams['font.size'] = 25
    # And a smaller font for the legend
    plt.rcParams['legend.fontsize'] = 18
    plt.rcParams['axes.labelsize'] = 35
    plt.rcParams['axes.titlesize'] = 35
    plt.rcParams['legend.labelspacing'] = 0.1
    # Shorter lines in the legend
    plt.rcParams['legend.handlelength'] = .7
    # Less space between legend entries
    plt.rcParams['legend.handletextpad'] = .25
    # Less space between lines and text in legend
    plt.rcParams['legend.borderpad'] = .1
    # Less space between lines of text in legend
    plt.rcParams['legend.borderaxespad'] = .5

    # And for the values on the axes
    plt.rcParams['xtick.labelsize'] = 25
    plt.rcParams['ytick.labelsize'] = 25
    # White background
    plt.rcParams['axes.facecolor'] = 'white'
    # Grey grid
    plt.rcParams['axes.grid'] = False
    #plt.rcParams['grid.color'] = 'grey'
    # Borders
    plt.rcParams['axes.linewidth'] = 2
    plt.rcParams['xtick.major.width'] = 2
    plt.rcParams['ytick.major.width'] = 2
    plt.rcParams['xtick.minor.width'] = 2
    plt.rcParams['ytick.minor.width'] = 2
    plt.rcParams['xtick.major.size'] = 10
    plt.rcParams['ytick.major.size'] = 10
    plt.rcParams['xtick.minor.size'] = 5
    plt.rcParams['ytick.minor.size'] = 5
    plt.rcParams['xtick.direction'] = 'out'
    plt.rcParams['ytick.direction'] = 'out'
    plt.rcParams['xtick.top'] = False #True
    plt.rcParams['ytick.right'] = False #True
    plt.rcParams['xtick.bottom'] = True
    plt.rcParams['ytick.left'] = True
    #plt.rcParams['xtick.minor.visible'] = True
    #plt.rcParams['ytick.minor.visible'] = True
    #plt.rcParams['xtick.color'] = 'black'
    #plt.rcParams['ytick.color'] = 'black'
    plt.rcParams['axes.edgecolor'] = 'black'
    plt.rcParams['axes.labelcolor'] = 'black'
    plt.rcParams.update(
        {
            "text.usetex": True,
            "text.latex.preamble": r"\usepackage{bm}",
            "text.latex.preamble": r"\usepackage{amsmath}",

            # Enforce default LaTeX font.
            #"font.family": "serif",
            #"font.serif": ["Computer Modern"],
        }
    )


