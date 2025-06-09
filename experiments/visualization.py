import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import LogLocator

import pandas as pd

import seaborn as sns


STANDARD_X_LIMS = (-0.15, 141)
STANDARD_Y_LIMS = (-0.02, 1.02)


def create_line_chart(df: pd.DataFrame, experiment_title: str, x_axis = 'token_size', y_axis = 'correct_guess', category = 'qa', n_types = 5, x_label = 'Token Size (Log$\mathregular{_2}$ Scale)', y_label = 'Accuracy', 
x_lims = STANDARD_X_LIMS, y_lims = STANDARD_Y_LIMS, y_grid = False, y_grid_minor = False, ylog = False):
    # ColorBrewer2 does not have colorblind friendly for 5 classes, so we trust in Seaborn/MatPlotLib instead
    fig, ax = plt.subplots(figsize=(10, 6))



    sns.lineplot(
        data=df,
        x=x_axis,
        y=y_axis,
        hue=category,
        palette=sns.color_palette('colorblind6', n_colors=n_types),
        marker='o'
    )

    # Set limits to allow space around plots
    if x_lims is not None:
        plt.xlim(x_lims[0], x_lims[1])
    if y_lims is not None:
        plt.ylim(y_lims[0], y_lims[1])

    # Set x-axis to logarithmic scale (log 2). Need it 'symlog' because we have a value for '0' (which is undefined in traditonal log).
    plt.xscale('symlog', base = 2)

    # Set y-axis
    if ylog:
        plt.yscale('log', base = 10)

    # Set major and minor locators for the x-axis. So we want the 'log-lines' in base 10. The major ones in base 2
    ax.xaxis.set_major_locator(LogLocator(2))


    # # Add grid lines
    ax.grid(True, which='major', axis='x', linestyle='-', linewidth=1)
    if y_grid:
        ax.grid(True, which='major', axis='y', linestyle='-', linewidth=0.5)
    
    if y_grid_minor:
        ax.grid(True, which='minor', axis = 'y', linestyle='--', linewidth = 0.25)


    # Get current xticks and rewrite them. Also add remove the minor x-ticks (for the log-lines)
    xticks_major = plt.xticks(minor=False)[0]

    xticks_major = [int(xtick) for xtick in xticks_major if (float(xtick) == 0 or float(xtick) >= 1)]
    xticks_major = xticks_major[:-2]
    xticks_major.insert(0, 0)

    # # Generate xtick labels
    xtick_labels = [f"{int(tick)}k" for tick in xticks_major]
    plt.xticks(xticks_major, xtick_labels, minor=False)

    # Remove top and right boxes
    plt.gca().spines["top"].set_visible(False)
    plt.gca().spines["right"].set_visible(False)
    ax.spines['bottom'].set_position(('outward', 10))

    plt.title(experiment_title, fontsize = 20, pad=43, weight=600)
    plt.xlabel(x_label, fontsize = 20, labelpad=10)
    plt.ylabel(y_label, fontsize = 20, labelpad=10)
    plt.legend(loc='upper center', fontsize = 16, bbox_to_anchor=(0.5, 1.14), ncols=5, frameon=False)
    plt.tick_params(axis='both', labelsize = 16)
    plt.tight_layout()
    plt.savefig(f"{experiment_title}-linechart.pdf")
    plt.show()


def create_heatmap(df: pd.DataFrame, experiment_title: str, index_type = 'qa', columns = 'token_size', values = 'correct_guess', x_label = 'Token Size', y_label = 'Question Type'):
    heatmap_data = df.pivot(index=index_type, columns=columns, values=values)

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = ['#ffffcc','#c2e699','#78c679','#31a354','#006837']
    colormap = LinearSegmentedColormap.from_list('YlGn', colors, N=256)

    sns.heatmap(
        heatmap_data,
        vmin=0,
        vmax=1,
        cmap=colormap,
        annot=True,
        annot_kws={"fontsize": 14},
        fmt=".2f",
        linewidths=0.5,
        linecolor='white'
    )

    # Generate xtick labels
    xtick_labels = ax.get_xticklabels(minor=False)
    xtick_labels = [f"{int(label.get_text())}k" for label in xtick_labels]
    ax.set_xticklabels(xtick_labels)

    plt.title(experiment_title, fontsize = 20, pad = 12, weight=600)
    plt.xlabel(x_label, fontsize = 20, labelpad=10)
    plt.ylabel(y_label, fontsize = 20, labelpad=10)
    ax.collections[0].colorbar.ax.tick_params(labelsize=16)
    plt.tick_params(axis='both', labelsize = 16)
    plt.tight_layout()
    plt.savefig(f"{experiment_title}-heatmap.pdf")
    plt.show()


def create_accuracy_figures(df: pd.DataFrame, experiment_title: str, groupby1 = 'qa', groupby2 = 'token_size', meanby = 'correct_guess'):
    accuracy_df = df.groupby(['qa', 'token_size'])['correct_guess'].mean().reset_index()
    accuracy_df['qa'] = accuracy_df['qa'].apply(lambda x: x.upper())

    create_heatmap(accuracy_df, experiment_title)
    create_line_chart(accuracy_df, experiment_title)
