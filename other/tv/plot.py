"""Plotting helpers for trait vector analysis.

heatmap: annotated matrix with diverging colormap (variants × traits, etc.)
bar_chart: horizontal bar chart with value annotations
similarity_matrix: NxN symmetric correlation/similarity heatmap
radar: polar/spider chart comparing trait profiles
grouped_bars: multiple series per category (e.g. text vs model delta)

All functions return a matplotlib Figure. Pass save="path.png" to save and close.
Requires matplotlib: pip install matplotlib
"""

import numpy as np

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
except ImportError:
    raise ImportError("plot module requires matplotlib: pip install matplotlib")


def _finish(fig, save):
    """Save figure if path given, otherwise leave open."""
    fig.tight_layout()
    if save:
        fig.savefig(save, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: {save}")
    return fig


def _text_color(val, vmax, threshold=0.55):
    """White text on dark cells, black on light."""
    return "white" if abs(val) > vmax * threshold else "black"


def heatmap(matrix, row_labels, col_labels, *, title=None, annotate=True,
            cmap="RdBu_r", symmetric=True, fmt=".2f", figsize=None, save=None):
    """Annotated heatmap with diverging colormap.

    Args:
        matrix: 2D numpy array [n_rows, n_cols]
        row_labels: list of row names
        col_labels: list of column names
        symmetric: if True, center colormap at zero (vmin=-vmax)
        fmt: annotation format string
        save: path to save figure (None = leave open)

    Returns:
        matplotlib Figure
    """
    matrix = np.asarray(matrix)
    n_rows, n_cols = matrix.shape

    if figsize is None:
        figsize = (max(8, n_cols * 0.6 + 3), max(4, n_rows * 0.5 + 1.5))
    fig, ax = plt.subplots(figsize=figsize)

    vmax = np.nanmax(np.abs(matrix)) if symmetric else np.nanmax(matrix)
    vmin = -vmax if symmetric else np.nanmin(matrix)

    im = ax.imshow(matrix, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")

    ax.set_xticks(range(n_cols))
    ax.set_xticklabels(col_labels, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(n_rows))
    ax.set_yticklabels(row_labels, fontsize=9)

    if annotate:
        for i in range(n_rows):
            for j in range(n_cols):
                val = matrix[i, j]
                if not np.isnan(val):
                    ax.text(j, i, f"{val:{fmt}}", ha="center", va="center",
                            fontsize=7, color=_text_color(val, vmax))

    fig.colorbar(im, ax=ax, shrink=0.8)
    if title:
        ax.set_title(title, fontsize=11, pad=10)

    return _finish(fig, save)


def bar_chart(labels, values, *, horizontal=True, title=None, colors=None,
              annotate=True, sort=True, figsize=None, save=None):
    """Bar chart with value annotations.

    Args:
        labels: list of bar names
        values: list of float values
        horizontal: horizontal bars (True) or vertical (False)
        sort: sort by |value| descending
        save: path to save figure

    Returns:
        matplotlib Figure
    """
    labels = list(labels)
    values = np.asarray(values)

    if sort:
        order = np.argsort(np.abs(values))[::-1]
        labels = [labels[i] for i in order]
        values = values[order]

    n = len(labels)
    if figsize is None:
        figsize = (10, max(3, n * 0.35)) if horizontal else (max(6, n * 0.5), 6)
    fig, ax = plt.subplots(figsize=figsize)

    if colors is None:
        colors = ["#d64541" if v >= 0 else "#3498db" for v in values]

    pos = range(n)
    if horizontal:
        ax.barh(pos, values, color=colors)
        ax.set_yticks(pos)
        ax.set_yticklabels(labels, fontsize=8)
        ax.invert_yaxis()
        ax.axvline(0, color="black", linewidth=0.5)
        if annotate:
            for i, v in enumerate(values):
                ax.text(v + (0.002 if v >= 0 else -0.002), i, f"{v:+.3f}",
                        va="center", ha="left" if v >= 0 else "right", fontsize=7)
    else:
        ax.bar(pos, values, color=colors)
        ax.set_xticks(pos)
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
        ax.axhline(0, color="black", linewidth=0.5)

    if title:
        ax.set_title(title, fontsize=11)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    return _finish(fig, save)


def similarity_matrix(matrix, labels, *, title=None, metric_name="cosine",
                      cmap="RdYlGn", figsize=None, save=None):
    """NxN symmetric correlation/similarity heatmap.

    Args:
        matrix: 2D numpy array [n, n] with values typically in [-1, 1]
        labels: list of variant/condition names
        metric_name: label for colorbar
        save: path to save figure

    Returns:
        matplotlib Figure
    """
    matrix = np.asarray(matrix)
    n = matrix.shape[0]

    if figsize is None:
        figsize = (max(5, n * 0.9 + 2), max(4, n * 0.8 + 1.5))
    fig, ax = plt.subplots(figsize=figsize)

    im = ax.imshow(matrix, cmap=cmap, vmin=-1, vmax=1, aspect="equal")

    ax.set_xticks(range(n))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=9)
    ax.set_yticks(range(n))
    ax.set_yticklabels(labels, fontsize=9)

    for i in range(n):
        for j in range(n):
            val = matrix[i, j]
            ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                    fontsize=8, color=_text_color(val, 1.0))

    cb = fig.colorbar(im, ax=ax, shrink=0.8)
    cb.set_label(metric_name)
    if title:
        ax.set_title(title, fontsize=11, pad=10)

    return _finish(fig, save)


def radar(profiles, labels, trait_names, *, colors=None, title=None,
          figsize=None, save=None):
    """Polar/spider chart comparing multiple trait profiles.

    Args:
        profiles: list of arrays, each [n_traits] (one per condition)
        labels: list of condition names (one per profile)
        trait_names: list of trait names (axis labels)
        save: path to save figure

    Returns:
        matplotlib Figure
    """
    n_traits = len(trait_names)
    angles = np.linspace(0, 2 * np.pi, n_traits, endpoint=False).tolist()
    angles += angles[:1]  # close the polygon

    if figsize is None:
        figsize = (8, 8)
    fig, ax = plt.subplots(figsize=figsize, subplot_kw=dict(polar=True))

    default_colors = ["#e74c3c", "#3498db", "#2ecc71", "#f39c12", "#9b59b6", "#1abc9c"]
    if colors is None:
        colors = default_colors

    for i, (profile, label) in enumerate(zip(profiles, labels)):
        vals = list(profile) + [profile[0]]
        color = colors[i % len(colors)]
        ax.plot(angles, vals, color=color, linewidth=1.5, label=label)
        ax.fill(angles, vals, color=color, alpha=0.08)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(trait_names, fontsize=8)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1), fontsize=9)
    if title:
        ax.set_title(title, fontsize=11, pad=20)

    return _finish(fig, save)


def grouped_bars(data, group_labels, series_labels, *, horizontal=True,
                 colors=None, title=None, xlabel=None, ylabel=None,
                 sort_by=None, figsize=None, save=None):
    """Grouped bar chart (multiple series per category).

    Args:
        data: [n_series, n_groups] array
        group_labels: list of group/category names (e.g. trait names)
        series_labels: list of series names (e.g. variant names)
        sort_by: series index to sort groups by (descending value). None = no sort.
        save: path to save figure

    Returns:
        matplotlib Figure
    """
    data = np.asarray(data)
    n_series, n_groups = data.shape

    if sort_by is not None:
        order = np.argsort(data[sort_by])[::-1]  # high → low
        data = data[:, order]
        group_labels = [group_labels[i] for i in order]

    bar_width = 0.8 / n_series

    default_colors = ["#e74c3c", "#3498db", "#2ecc71", "#f39c12", "#9b59b6", "#1abc9c"]
    if colors is None:
        colors = default_colors

    if figsize is None:
        figsize = (10, max(3, n_groups * 0.4)) if horizontal else (max(6, n_groups * 0.6), 6)
    fig, ax = plt.subplots(figsize=figsize)

    for s in range(n_series):
        offsets = np.arange(n_groups) + s * bar_width - (n_series - 1) * bar_width / 2
        color = colors[s % len(colors)]
        if horizontal:
            ax.barh(offsets, data[s], height=bar_width, label=series_labels[s], color=color)
        else:
            ax.bar(offsets, data[s], width=bar_width, label=series_labels[s], color=color)

    if horizontal:
        ax.set_yticks(range(n_groups))
        ax.set_yticklabels(group_labels, fontsize=8)
        ax.invert_yaxis()
        ax.axvline(0, color="black", linewidth=0.5)
    else:
        ax.set_xticks(range(n_groups))
        ax.set_xticklabels(group_labels, rotation=45, ha="right", fontsize=8)
        ax.axhline(0, color="black", linewidth=0.5)

    ax.legend(fontsize=9)
    if title:
        ax.set_title(title, fontsize=11)
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=9)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    return _finish(fig, save)


def trajectory(x, lines, *, vlines=None, xlabel=None, ylabel=None,
               title=None, subtitle=None, figsize=None, ax=None, save=None):
    """Line plot with optional ±std bands and vertical markers.

    Args:
        x: array of x positions
        lines: [{y, std (optional), label, color}]
        vlines: [{x, label, color (optional), style (optional)}]
        ax: existing axes to plot on (for subplots). If None, creates new figure.
        save: path to save figure (only used when ax is None)

    Returns:
        matplotlib Figure (or ax if ax was provided)
    """
    x = np.asarray(x)
    own_fig = ax is None

    if own_fig:
        if figsize is None:
            figsize = (10, 5)
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    for line in lines:
        y = np.asarray(line["y"])
        color = line.get("color")
        label = line.get("label")
        ax.plot(x, y, color=color, label=label, linewidth=1.5)
        if "std" in line:
            std = np.asarray(line["std"])
            ax.fill_between(x, y - std, y + std, color=color, alpha=0.15)

    if vlines:
        for vl in vlines:
            style = vl.get("style", "--")
            color = vl.get("color", "gray")
            ax.axvline(vl["x"], linestyle=style, color=color, linewidth=1, alpha=0.7)
            if "label" in vl:
                ax.text(vl["x"] + 0.5, ax.get_ylim()[1] * 0.95, vl["label"],
                        fontsize=7, color=color, va="top")

    ax.axhline(0, color="black", linewidth=0.3, alpha=0.5)
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=9)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=9)
    if title:
        ax.set_title(title, fontsize=11)
    if subtitle:
        ax.set_title(subtitle, fontsize=8, loc="left", style="italic")
    ax.legend(fontsize=8, loc="best")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    if own_fig:
        return _finish(fig, save)
    return ax


def trajectory_grid(x, panels, *, vlines=None, ncols=4, xlabel=None,
                    figsize=None, save=None):
    """Grid of trajectory subplots (e.g. per-trait onset dynamics).

    Args:
        x: array of x positions (shared across panels)
        panels: [{title, lines: [{y, std, label, color}]}]
        vlines: [{x, label, color, style}] — applied to all panels
        ncols: columns in grid
        save: path to save figure

    Returns:
        matplotlib Figure
    """
    n = len(panels)
    nrows = (n + ncols - 1) // ncols
    if figsize is None:
        figsize = (ncols * 3.5, nrows * 2.5)
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, sharex=True)
    axes = np.atleast_2d(axes)

    for i, panel in enumerate(panels):
        r, c = divmod(i, ncols)
        ax = axes[r, c]
        trajectory(x, panel["lines"], vlines=vlines, ax=ax,
                   title=panel.get("title"))

    # Hide unused axes
    for i in range(n, nrows * ncols):
        r, c = divmod(i, ncols)
        axes[r, c].set_visible(False)

    if xlabel:
        for c in range(min(ncols, n)):
            axes[-1, c].set_xlabel(xlabel, fontsize=8)

    fig.tight_layout()
    return _finish(fig, save)
