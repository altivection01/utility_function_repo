# ECDF plotting utility aligned with existing boxplot style.
# This is a more robust way to visualize the distribution of delay metrics, especially those with heavy tails and many zeros.
# The function takes in the column to plot, the dataframe, an optional hue for grouping, and options for title and axis labels.
def ecdfplot(
    x,
    data,
    hue=None,
    title=None,
    xlabel=None,
    ylabel="Fraction of Flights",
    log_x=False
):

    # Match figure scale with boxplot utility
    plt.figure(figsize=(18, 10))

    # ECDF plot
    sns.ecdfplot(
        data=data,
        x=x,
        hue=hue,
        linewidth=2
    )

    # Axis labels and title
    if title:
        plt.title(title, fontsize=16)
    if xlabel:
        plt.xlabel(xlabel, fontsize=14)
    plt.ylabel(ylabel, fontsize=14)

    # Optional log scale for heavy tails
    if log_x:
        plt.xscale("symlog")  # preserves zeros, compresses tail

    # Grid for percentile readability
    plt.grid(True, which="both", linestyle="--", alpha=0.4)

    # Layout polish
    plt.tight_layout()

    plt.show()
