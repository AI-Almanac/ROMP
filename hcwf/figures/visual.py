"""Plotting helpers: monthly onset colormap, map basemaps and boxes, and portrait/heatmap plots.

The shared building blocks of the figure modules of `hcwf.figures`: the season colormap (`cbar_season`), the region
basemap and outlines (`set_basemap`, `shp_outline`, `get_india_outline`, `polygon_outline`, `add_polygon`,
`box_boundary`) and the portrait plot (`portrait_plot` with `prepare_data`, `heatmap`, `annotate_heatmap` and
`add_legend`), a simplified version of the PCMDI Metrics Package (PMP) portrait plot.
"""
import sys

import matplotlib
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
import numpy as np
import logging

log = logging.getLogger(__name__)


def cbar_season():
    """Create the custom monthly colormap (May to September).

    Each month has its own sequential colormap, sliced into 8 steps; the steps are the colours of the days of year
    121 to 273.

    Returns
    -------
    cmap_jjas : matplotlib.colors.ListedColormap
        The colormap of 40 colours.
    norm_jjas : matplotlib.colors.BoundaryNorm
        The norm mapping day of year onto `cmap_jjas`.
    bounds : list of float
        The 41 boundaries of the norm, in day of year.
    """
    from matplotlib.colors import ListedColormap, BoundaryNorm

    month_cmaps = {
    "May":    plt.cm.YlOrBr,
    "Jun":    plt.cm.Greens,
    "Jul":    plt.cm.YlGnBu,
    "Aug":    plt.cm.Purples,
    "Sep":    plt.cm.RdPu,
    }

    # Day-of-year ranges (only until September)
    month_doys = {
        "May": (121, 151),  # May 15 - May 31
        "Jun": (152, 181),  # Jun 1 - Jun 30
        "Jul": (182, 212),  # Jul 1 - Jul 31
        "Aug": (213, 243),  # Aug 1 - Aug 31
        "Sep": (244, 273),  # Sep 1 - Sep 30
    }
    colors = []
    bounds = []

    N_per_month = 8  # smoothness inside each month

    for month, cmap in month_cmaps.items():
        d0, d1 = month_doys[month]

        # Sequential slice of each colormap
        ramp = cmap(np.linspace(0.35, 0.95, N_per_month))

        colors.extend(ramp)
        bounds.extend(np.linspace(d0, d1, N_per_month, endpoint=False))

    # Add final bound
    bounds.append(list(month_doys.values())[-1][1])

    cmap_jjas = ListedColormap(colors, name="JJAS_piecewise")
    norm_jjas = BoundaryNorm(bounds, cmap_jjas.N)

    return cmap_jjas, norm_jjas, bounds


def set_basemap(ax, spec):
    """Draw the region outline, set the map extent and add the gridlines of a Cartopy axis.

    Parameters
    ----------
    ax : cartopy.mpl.geoaxes.GeoAxes
        The axes to draw on.
    spec : object
        The run's ``FigureSpec``: its ``region`` gives the extent (and the outline), and ``shpfile_dir`` (a
        shapefile path) the outline when set.

    Returns
    -------
    ax : cartopy.mpl.geoaxes.GeoAxes
        The axes.
    gl : cartopy.mpl.gridliner.Gridliner
        The gridliner, with labels on the left and bottom, every 5 degrees.
    """
    region, shpfile_dir = spec.region, spec.shpfile_dir
    from hcwf.onset.regions import box
    import cartopy.crs as ccrs
    import matplotlib.ticker as mticker

    # Add Ethiopia boundary
    if shpfile_dir:
        region_boundaries = get_india_outline(shpfile_dir)
        for boundary in region_boundaries:
            region_lon, region_lat = boundary
            ax.plot(region_lon, region_lat, color='black', linewidth=1.5)
    else:
        shp_outline(ax, region=region)

    # Set map extent to Ethiopia
    lats, latn, lonw, lone = box(region)
    ax.set_extent([lonw, lone, lats, latn], crs=ccrs.PlateCarree())

    gl = ax.gridlines(draw_labels=True, dms=False, x_inline=False, y_inline=False,
                      linewidth=0, linestyle='', color='none', alpha=0)

    gl.top_labels = False
    gl.right_labels = False
    gl.xlabel_style = {'size': 7}
    gl.ylabel_style = {'size': 7}

    gl.xlocator = mticker.MultipleLocator(5)   # every 5°
    gl.ylocator = mticker.MultipleLocator(5)   # every 5°

    return ax, gl

def box_boundary(rect_boundary, ax, *, edgecolor='black', linewidth=2,
                    linestyle='-', fill=False, alpha=1.0,
                    zorder=10):
    """Draw a lat/lon rectangle on an axes (in PlateCarree coordinates on a Cartopy GeoAxes).

    Nothing is drawn for the placeholder name ``"rect_boundary"`` or when the region has coordinates outside
    ``[-180, 360]``.

    Parameters
    ----------
    rect_boundary : str
        The region name whose box is drawn (see `hcwf.onset.regions.box`).
    ax : matplotlib.axes.Axes
        The axes to draw on.
    edgecolor : str, default "black"
        The edge colour.
    linewidth : float, default 2
        The edge width.
    linestyle : str, default "-"
        The edge line style.
    fill : bool, default False
        Fill the rectangle.
    alpha : float, default 1.0
        The opacity.
    zorder : float, default 10
        The drawing order.

    Returns
    -------
    matplotlib.axes.Axes
        `ax` (the rectangle is added to it).
    """
    import matplotlib.patches as mpatches
    from hcwf.onset.regions import box

    if rect_boundary == "rect_boundary":  # the plots' placeholder name: no box is drawn until a region is given
        return ax
    lats, latn, lonw, lone = box(rect_boundary)

    vals = [lats, latn, lonw, lone]

    if any(v > 360 or v < -180 for v in vals):
        return ax

    # Normalize bounds (safety)
    lonw, lone = sorted([lonw, lone])
    lats, latn = sorted([lats, latn])

    # Detect axes type
    transform = ax.transData  # default: data coords

    try:
        import cartopy.mpl.geoaxes as cgeo

        if isinstance(ax, cgeo.GeoAxes):
            import cartopy.crs as ccrs
            transform = ccrs.PlateCarree()
    except ImportError:
        # Cartopy not installed → silently fall back to data coords
        pass

    rect = mpatches.Rectangle(
        (lonw, lats),
        lone - lonw,
        latn - lats,
        fill=fill,
        edgecolor=edgecolor,
        linewidth=linewidth,
        linestyle=linestyle,
        alpha=alpha,
        transform=transform,
        zorder=zorder
    )

    ax.add_patch(rect)
    return ax


def portrait_plot(
    data,
    xaxis_labels,
    yaxis_labels,
    fig=None,
    ax=None,
    annotate=False,
    annotate_data=None,
    annotate_textcolors=("black", "white"),
    annotate_textcolors_threshold=None,
    annotate_fontsize=15,
    annotate_format="{x:.2f}",
    figsize=(12, 10),
    vrange=None,
    xaxis_fontsize=12,
    yaxis_fontsize=12,
    xaxis_tick_labels_top_and_bottom=False,
    xticklabel_rotation=45,
    title="",
    cmap="RdBu_r",
    cmap_bounds=None,
    cbar_label=None,
    cbar_label_fontsize=15,
    cbar_tick_fontsize=12,
    cbar_kw={},
    colorbar_off=False,
    missing_color="grey",
    invert_yaxis=True,
    box_as_square=False,
    legend_on=False,
    legend_labels=None,
    legend_box_xy=None,
    legend_box_size=None,
    legend_lw=1,
    legend_fontsize=14,
    debug=False,
):
    """Draw a portrait plot: a heatmap of a 2D array with labelled rows and columns.

    Only the heatmap style (a single 2D array) is drawn; stacked arrays (2 or 4, meant as triangles) pass the
    checks but draw nothing and fail at the return. The function exits the program with a message (`sys.exit`)
    when the inputs are inconsistent. This is a simplified version of the PMP ``portrait_plot()``.

    Parameters
    ----------
    data : numpy.ndarray or list of numpy.ndarray
        A 2D array, a list of 2D arrays, or a 3D array (stacked 2D arrays), with shape (row, column).
    xaxis_labels : list of str
        The column labels; the length equals the number of columns, or is 0 to turn off the labels.
    yaxis_labels : list of str
        The row labels; the length equals the number of rows, or is 0 to turn off the labels.
    fig : matplotlib.figure.Figure, optional
        The figure to plot on. When None a new figure is created.
    ax : matplotlib.axes.Axes, optional
        The axes to plot on. When None a new subplot of `fig` is created.
    annotate : bool, default False
        Add the value as text in each cell (heatmap style only).
    annotate_data : numpy.ndarray, optional
        The values written in the cells. None uses `data`.
    annotate_textcolors : tuple of (str, str), default ("black", "white")
        The text colours below and above `annotate_textcolors_threshold`.
    annotate_textcolors_threshold : tuple or float, optional
        The value in data units where the text colour switches. Currently not passed on (see `annotate_heatmap`).
    annotate_fontsize : float, default 15
        The font size of the annotations (divided by half the number of columns).
    annotate_format : str, default "{x:.2f}"
        The format of the annotated values.
    figsize : tuple of (float, float), default (12, 10)
        The figure size in inches, when a figure is created.
    vrange : tuple of (float, float), optional
        The range of the colour scale. None uses the minimum and maximum of `data`.
    xaxis_fontsize : float, default 12
        The font size of the column tick labels (and the title).
    yaxis_fontsize : float, default 12
        The font size of the row tick labels.
    xaxis_tick_labels_top_and_bottom : bool, default False
        Put the column tick labels on top and bottom instead of the bottom only.
    xticklabel_rotation : float, default 45
        The rotation of the column tick labels, in degrees.
    title : str, default ""
        The title of the axes.
    cmap : str, default "RdBu_r"
        The matplotlib colormap name.
    cmap_bounds : list of float, optional
        Boundaries of discrete colours. None gives a continuous scale.
    cbar_label : str, optional
        The colorbar label.
    cbar_label_fontsize : float, default 15
        The font size of the colorbar label.
    cbar_tick_fontsize : float, default 12
        The font size of the colorbar tick labels.
    cbar_kw : dict, default {}
        Keywords of `matplotlib.figure.Figure.colorbar`.
    colorbar_off : bool, default False
        Draw no colorbar.
    missing_color : str, default "grey"
        The background colour, seen at NaN cells.
    invert_yaxis : bool, default True
        Put the first row at the top.
    box_as_square : bool, default False
        Make each cell square.
    legend_on : bool, default False
        Show a legend (for 2 or 4 triangles only).
    legend_labels : list of str, optional
        The legend labels; required when `legend_on` is true.
    legend_box_xy : tuple of (float, float), optional
        The upper-left corner of the legend box in axes coordinates (lower-left if `invert_yaxis` is false).
    legend_box_size : float, optional
        The size of the legend box.
    legend_lw : float, default 1
        The line width of the legend.
    legend_fontsize : float, default 14
        The font size of the legend.
    debug : bool, default False
        Log more messages.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure.
    ax : matplotlib.axes.Axes
        The axes.
    im : matplotlib.collections.QuadMesh
        The drawn mesh, for a common colorbar.
    cbar : matplotlib.colorbar.Colorbar
        The colorbar; returned (as a fourth value) only when `colorbar_off` is false.
    """
    # ----------------
    # Prepare plotting
    # ----------------
    data, num_divide = prepare_data(data, xaxis_labels, yaxis_labels, debug=debug)

    if num_divide not in [1, 2, 4]:
        sys.exit("Error: Number of (stacked) array is not 1, 2, or 4.")

    if annotate:
        if annotate_data is None:
            annotate_data = data
            num_divide_annotate = num_divide
        else:
            annotate_data, num_divide_annotate = prepare_data(
                annotate_data, xaxis_labels, yaxis_labels, debug=debug
            )
            if num_divide_annotate != num_divide:
                sys.exit("Error: annotate_data does not have same size as data")

    # ----------------
    # Ready to plot!!
    # ----------------
    if fig is None:
        fig = plt.figure(figsize=figsize)
    if ax is None:
        ax = fig.add_subplot(111)

    ax.set_facecolor(missing_color)

    if vrange is None:
        vmin = np.nanmin(data)
        vmax = np.nanmax(data)
    else:
        vmin = min(vrange)
        vmax = max(vrange)

    # Normalize colorbar
    if cmap_bounds is None:
        norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)
    else:
        cmap = plt.get_cmap(cmap)
        if "extend" in list(cbar_kw.keys()):
            extend = cbar_kw["extend"]
        else:
            extend = "neither"
        norm = matplotlib.colors.BoundaryNorm(cmap_bounds, cmap.N, extend=extend)

    # [1] Heatmap-style portrait plot (no triangles)
    if num_divide == 1:
        ax, im = heatmap(
            data,
            xaxis_labels,
            yaxis_labels,
            ax=ax,
            invert_yaxis=invert_yaxis,
            cmap=cmap,
            edgecolors="k",
            linewidth=0.5,
            norm=norm,
        )
        if annotate:
            if annotate_data is not None:
                if annotate_data.shape != data.shape:
                    sys.exit("Error: annotate_data has different size than data")
            else:
                annotate_data = data
            ax = annotate_heatmap(
                im,
                ax=ax,
                data=data,
                cmap=cmap,
                annotate_data=annotate_data,
                valfmt=annotate_format,
                textcolors=annotate_textcolors,
                threshold=annotate_textcolors_threshold,
                fontsize=annotate_fontsize,
            )

    # X-axis tick labels
    if xaxis_tick_labels_top_and_bottom:
        # additional x-axis tick labels
        ax.tick_params(axis="x", bottom=True, top=True, labelbottom=True, labeltop=True)
    else:
        # Let the horizontal axes labeling appear on bottom.
        ax.tick_params(top=False, bottom=True, labeltop=False, labelbottom=True)

    # Rotate and align top ticklabels
    plt.setp(
        [tick.label2 for tick in ax.xaxis.get_major_ticks()],
        rotation=xticklabel_rotation,
        ha="left",
        va="center",
        rotation_mode="anchor",
        fontsize=xaxis_fontsize,
    )

    if xaxis_tick_labels_top_and_bottom:
        # Rotate and align bottom ticklabels
        plt.setp(
            [tick.label1 for tick in ax.xaxis.get_major_ticks()],
            rotation=xticklabel_rotation,
            ha="right",
            va="center",
            rotation_mode="anchor",
            fontsize=xaxis_fontsize,
        )

    # Set font size for yaxis tick labels
    plt.setp(ax.get_yticklabels(), fontsize=yaxis_fontsize)

    # Legend
    if legend_on:
        if legend_labels is None:
            sys.exit("Error: legend_labels was not provided.")
        else:
            add_legend(
                num_divide,
                ax,
                legend_box_xy,
                legend_box_size,
                labels=legend_labels,
                lw=legend_lw,
                fontsize=legend_fontsize,
            )

    if title:
        ax.set_title(title, fontsize=xaxis_fontsize, fontweight='normal')

    if box_as_square:
        ax.set_aspect("equal")

    log.debug("%s %s", "im = ", im)
    if not colorbar_off:
        # Create colorbar
        cbar_pad = 0.14
        cbar_aspect = 16
        cbar_shrink = 0.75
        cbar = ax.figure.colorbar(im, ax=ax, orientation='horizontal',
                                  pad=cbar_pad, shrink=cbar_shrink, aspect=cbar_aspect, **cbar_kw)

        # Label for colorbar
        if cbar_label is not None:
            if "orientation" in list(cbar_kw.keys()):
                if cbar_kw["orientation"] == "vertical":
                    rotation = -90
                    ha = "center"
                    va = "bottom"
                    cbar.ax.set_xlabel(
                        cbar_label,
                        rotation=rotation,
                        ha=ha,
                        va=va,
                        fontsize=cbar_label_fontsize,
                    )
                else:
                    rotation = 0
                    ha = "center"
                    va = "top"
                    cbar.ax.set_ylabel(
                        cbar_label,
                        rotation=rotation,
                        ha=ha,
                        va=va,
                        fontsize=cbar_label_fontsize,
                    )
            else:
                rotation = 0
                ha = "center"
                va = "top"
                cbar.ax.set_ylabel(
                    cbar_label,
                    rotation=rotation,
                    ha=ha,
                    va=va,
                    fontsize=cbar_label_fontsize,
                )
            cbar.ax.tick_params(labelsize=cbar_tick_fontsize)
        return fig, ax, im, cbar
    else:
        return fig, ax, im


# ======================================================================
# Prepare data
# ----------------------------------------------------------------------
def prepare_data(data, xaxis_labels, yaxis_labels, debug=False):
    """Convert `data` to a numpy array and check its shape against the labels.

    The program exits (`sys.exit`) with a message when the data are not a 2D or 3D array, a list that is not made
    of 2D arrays, or when the labels do not match the shape (an empty label list matches anything).

    Parameters
    ----------
    data : numpy.ndarray or list of numpy.ndarray
        A 2D array, a list of 2D arrays, or a 3D array of stacked 2D arrays, with shape (row, column).
    xaxis_labels : list of str
        The column labels.
    yaxis_labels : list of str
        The row labels.
    debug : bool, default False
        Log the type and shape of the data.

    Returns
    -------
    data : numpy.ndarray
        The data as an array, shape (row, column) or (stack, row, column).
    num_divide : int
        The number of stacked 2D arrays (1 for a single 2D array).
    """
    # In case data was given as list of arrays, convert it to numpy (stacked) array
    if isinstance(data, list):
        if debug:
            log.debug("data type is list")
            log.debug("%s %s", "len(data):", len(data))
        if len(data) == 1:  # list has only 1 array as element
            if isinstance(data[0], np.ndarray) and (len(data[0].shape) == 2):
                data = data[0]
                num_divide = 1
            else:
                sys.exit("Error: Element of given list is not in np.ndarray type")
        else:  # list has more than 1 arrays as elements
            data = np.stack(data)
            num_divide = len(data)

    # Now, data is expected to be be a numpy array (whether given or converted from list)
    if debug:
        log.debug("%s %s", "data.shape:", data.shape)

    if data.shape[-1] != len(xaxis_labels) and len(xaxis_labels) > 0:
        sys.exit("Error: Number of elements in xaxis_label mismatchs to the data")

    if data.shape[-2] != len(yaxis_labels) and len(yaxis_labels) > 0:
        sys.exit("Error: Number of elements in yaxis_label mismatchs to the data")

    if isinstance(data, np.ndarray):
        if len(data.shape) == 2:
            num_divide = 1
        elif len(data.shape) == 3:
            num_divide = data.shape[0]
        else:
            log.debug("%s %s", "data.shape:", data.shape)
            sys.exit("Error: data.shape is not right")
    else:
        sys.exit("Error: Converted or given data is not in np.ndarray type")

    if debug:
        log.debug("%s %s", "num_divide:", num_divide)

    return data, num_divide


# ======================================================================
# Portrait plot 1: heatmap-style (no triangle)
# (Inspired from: https://matplotlib.org/devdocs/gallery/images_contours_and_fields/image_annotated_heatmap.html)
# ----------------------------------------------------------------------
def heatmap(data, xaxis_labels, yaxis_labels, ax=None, invert_yaxis=False, **kwargs):
    """Create a heatmap from a numpy array and two lists of labels.

    Parameters
    ----------
    data : numpy.ndarray, shape (M, N)
        The values.
    xaxis_labels : list of str
        The N labels of the columns.
    yaxis_labels : list of str
        The M labels of the rows.
    ax : matplotlib.axes.Axes, optional
        The axes to plot on. None uses the current axes.
    invert_yaxis : bool, default False
        Put the first row at the top instead of the bottom.
    **kwargs : dict
        Passed on to `matplotlib.axes.Axes.pcolormesh`.

    Returns
    -------
    ax : matplotlib.axes.Axes
        The axes.
    im : matplotlib.collections.QuadMesh
        The drawn mesh.
    """
    if ax is None:
        ax = plt.gca()

    if invert_yaxis:
        ax.invert_yaxis()

    # Plot the heatmap
    im = ax.pcolormesh(data, **kwargs)

    # Show all ticks and label them with the respective list entries.
    ax.set_xticks(np.arange(data.shape[1]) + 0.5, minor=False)
    ax.set_yticks(np.arange(data.shape[0]) + 0.5, minor=False)
    ax.set_xticklabels(xaxis_labels)
    ax.set_yticklabels(yaxis_labels)
    ax.tick_params(which="minor", bottom=False, left=False)

    return ax, im


def annotate_heatmap(
    im,
    ax,
    data=None,
    cmap=None,
    annotate_data=None,
    valfmt="{x:.2f}",
    textcolors=("black", "white"),
    threshold=None,
    **textkw,
):
    """Annotate a heatmap with the value of each cell.

    The text colour is black or white, whichever contrasts with the cell colour (`contrast_text_color`, against a
    norm from the 10 % and 90 % quantiles of `data`); with a tuple `threshold` it is picked from `textcolors`
    instead. The font size is ``fontsize * 2 / number of columns``, so `fontsize` must be given in `textkw`.

    Parameters
    ----------
    im : matplotlib.collections.QuadMesh
        The drawn heatmap to label. Its array is used when `data` is not a list or array.
    ax : matplotlib.axes.Axes
        The axes of the heatmap.
    data : numpy.ndarray, shape (row, column), optional
        The data the image is coloured with.
    cmap : str or matplotlib.colors.Colormap, optional
        The colormap of the image.
    annotate_data : numpy.ndarray, shape (row, column), optional
        The values written in the cells. None uses `data`.
    valfmt : str or matplotlib.ticker.Formatter, default "{x:.2f}"
        The format of the annotations: a `str.format` string with ``x``, or a formatter.
    textcolors : tuple of (str, str), default ("black", "white")
        The colours of values inside and outside a tuple `threshold` (``(low, high)``).
    threshold : tuple of (float, float) or float, optional
        With a tuple, the cells outside ``[min, max]`` get the second colour. Otherwise unused for the colour (the
        contrast rule applies).
    **textkw : dict
        Passed on to `matplotlib.axes.Axes.text` for every cell; must contain ``fontsize``.

    Returns
    -------
    matplotlib.axes.Axes
        `ax`, with a text per cell.
    """
    if not isinstance(data, (list, np.ndarray)):
        data = im.get_array().reshape(im._meshHeight, im._meshWidth)

    if annotate_data is None:
        annotate_data = data

    if threshold is None:
        threshold = (data.max()) / 2.0

    # Set default alignment to center, but allow it to be
    # overwritten by textkw.
    kw = dict(horizontalalignment="center", verticalalignment="center")
    kw.update(textkw)

    # adjust annote fontsize according to # of heat blocks
    x_num = data.shape[1]
    fontsize = kw.get('fontsize')
    fontsize *= 2/x_num
    kw.update({'fontsize': fontsize})

    # Get the formatter in case a string is supplied
    if isinstance(valfmt, str):
        valfmt = matplotlib.ticker.StrMethodFormatter(valfmt)

    # Loop over the data and create a `Text` for each "pixel".
    # Change the text's color depending on the data.

    # determin text color based on background color luminance
    vmin, vmax = np.nanquantile(data, [0.1, 0.9])
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
    cmap_obj = plt.get_cmap(cmap)

    texts = []
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            if type(threshold) is tuple:
                kw.update(
                    color=textcolors[
                        int(
                            (data[i, j] > max(threshold))
                            or (data[i, j] < min(threshold))
                        )
                    ]
                )
            else:
                text_color = contrast_text_color(data[i,j], cmap_obj, norm, threshold=0.5)
                kw.update(color=text_color)

            text = ax.text(j + 0.5, i + 0.5, valfmt(annotate_data[i, j], None), **kw)
            texts.append(text)

    return ax


def contrast_text_color(value, cmap_obj, norm, threshold=0.5):
    """Return black or white, whichever contrasts better with the colormap colour at a value.

    Parameters
    ----------
    value : float
        The data value of the cell.
    cmap_obj : matplotlib.colors.Colormap
        The colormap.
    norm : matplotlib.colors.Normalize
        The norm mapping `value` to the colormap.
    threshold : float, default 0.5
        The luminance (WCAG perceived luminance, 0 to 1) above which the background counts as light.

    Returns
    -------
    str
        ``"black"`` on a light background, ``"white"`` on a dark one.
    """
    r, g, b, _ = cmap_obj(norm(value))  # background color
    luminance = 0.2126*r + 0.7152*g + 0.0722*b  # perceived luminance (WCAG standard)
    return 'black' if luminance > threshold else 'white'


# ======================================================================
# Portrait plot legend (four/two triangles)
# ======================================================================
def add_legend(
    num_divide, ax, box_xy=None, box_size=None, labels=None, lw=1, fontsize=14
):
    """Draw a legend box divided into 2 or 4 labelled triangles next to the portrait plot.

    Parameters
    ----------
    num_divide : int
        The number of triangles: 2 or 4 (anything else draws nothing).
    ax : matplotlib.axes.Axes
        The axes of the portrait plot.
    box_xy : tuple of (float, float), optional
        The upper-left corner of the box in axes coordinates. None puts it right of the axes.
    box_size : float, optional
        The size of the box in data units; default 1.5.
    labels : list of str, optional
        The triangle labels; defaults ``["TOP", "RIGHT", "BOTTOM", "LEFT"]`` (4) or ``["UPPER", "LOWER"]`` (2).
    lw : float, default 1
        The line width of the box.
    fontsize : float, default 14
        The font size of the labels.
    """
    if box_xy is None:
        box_x = ax.get_xlim()[1] * 1.25
        box_y = ax.get_ylim()[1]
    else:
        # Convert axes coordinate to data coordinate
        # Ref: https://matplotlib.org/stable/tutorials/advanced/transforms_tutorial.html
        box_x, box_y = ax.transLimits.inverted().transform(box_xy)

    if box_size is None:
        box_size = 1.5

    if num_divide == 4:
        if labels is None:
            labels = ["TOP", "RIGHT", "BOTTOM", "LEFT"]
        ax.add_patch(
            plt.Polygon(
                [
                    [box_x, box_y],
                    [box_x + box_size / 2.0, box_y + box_size / 2],
                    [box_x + box_size, box_y],
                ],
                color="k",
                fill=False,
                clip_on=False,
                lw=lw,
            )
        )
        ax.add_patch(
            plt.Polygon(
                [
                    [box_x + box_size, box_y],
                    [box_x + box_size / 2.0, box_y + box_size / 2],
                    [box_x + box_size, box_y + box_size],
                ],
                color="k",
                fill=False,
                clip_on=False,
                lw=lw,
            )
        )
        ax.add_patch(
            plt.Polygon(
                [
                    [box_x + box_size, box_y + box_size],
                    [box_x + box_size / 2.0, box_y + box_size / 2],
                    [box_x, box_y + box_size],
                ],
                color="k",
                fill=False,
                clip_on=False,
                lw=lw,
            )
        )
        ax.add_patch(
            plt.Polygon(
                [
                    [box_x, box_y],
                    [box_x + box_size / 2.0, box_y + box_size / 2],
                    [box_x, box_y + box_size],
                ],
                color="k",
                fill=False,
                clip_on=False,
                lw=lw,
            )
        )
        ax.text(
            box_x + box_size * 0.5,
            box_y + box_size * 0.2,
            labels[0],
            ha="center",
            va="center",
            fontsize=fontsize,
        )
        ax.text(
            box_x + box_size * 0.8,
            box_y + box_size * 0.5,
            labels[1],
            ha="center",
            va="center",
            fontsize=fontsize,
        )
        ax.text(
            box_x + box_size * 0.5,
            box_y + box_size * 0.8,
            labels[2],
            ha="center",
            va="center",
            fontsize=fontsize,
        )
        ax.text(
            box_x + box_size * 0.2,
            box_y + box_size * 0.5,
            labels[3],
            ha="center",
            va="center",
            fontsize=fontsize,
        )
    elif num_divide == 2:
        if labels is None:
            labels = ["UPPER", "LOWER"]
        ax.add_patch(
            plt.Polygon(
                [[box_x, box_y], [box_x, box_y + box_size], [box_x + box_size, box_y]],
                color="k",
                fill=False,
                clip_on=False,
                lw=lw,
            )
        )
        ax.add_patch(
            plt.Polygon(
                [
                    [box_x + box_size, box_y + box_size],
                    [box_x, box_y + box_size],
                    [box_x + box_size, box_y],
                ],
                color="k",
                fill=False,
                clip_on=False,
                lw=lw,
            )
        )
        ax.text(
            box_x + box_size * 0.05,
            box_y + box_size * 0.2,
            labels[0],
            ha="left",
            va="center",
            fontsize=fontsize,
        )
        ax.text(
            box_x + box_size * 0.95,
            box_y + box_size * 0.8,
            labels[1],
            ha="right",
            va="center",
            fontsize=fontsize,
        )


# ----------------------------------------------------------------------------------------------------------
# region outlines on maps
# ----------------------------------------------------------------------------------------------------------

def polygon_outline(ax, polygon1_lon, polygon1_lat, linewidth=1.25):
    """Add a polygon boundary to a basemap.

    The polygon is added as a feature with a high zorder so that it appears above the gridlines of Cartopy.

    Parameters
    ----------
    ax : cartopy.mpl.geoaxes.GeoAxes
        The axes to draw on.
    polygon1_lon : sequence of float
        Longitudes of the polygon vertices.
    polygon1_lat : sequence of float
        Latitudes of the polygon vertices.
    linewidth : float, default 1.25
        The outline width.

    Returns
    -------
    cartopy.mpl.geoaxes.GeoAxes
        `ax`.
    """
    import cartopy.crs as ccrs
    from shapely.geometry import Polygon as ShapelyPolygon
    from cartopy.feature import ShapelyFeature

    # Cartopy gridlines create a separate layer that can cover patches added with ax.add_patch()
    # use a ShapelyFeature and add it with ax.add_feature() with a higher zorder
    # so it appears above gridlines and map features
    poly_geom = ShapelyPolygon(list(zip(polygon1_lon, polygon1_lat)))

    # wrap as a Cartopy feature
    poly_feature = ShapelyFeature(
        [poly_geom],
        crs=ccrs.PlateCarree(),  # coordinates are lon/lat
        edgecolor='black',
        facecolor='none',
        linewidth=linewidth
    )

    # add to axes above gridlines
    ax.add_feature(poly_feature, zorder=10)

    return ax



def add_polygon(ax, da, polygon, return_polygon=False, linewidth=1.25):
    """Draw the polygon boundary for a field on `ax` when `polygon` is truthy and a boundary is defined.

    Parameters
    ----------
    ax : cartopy.mpl.geoaxes.GeoAxes
        The axes to draw on.
    da : xarray.DataArray
        The field; the boundary depends on its grid (see `hcwf.onset.regions.polygon_boundary`).
    polygon : bool
        Draw the boundary; with a false value nothing is drawn (and `return_polygon` must be false).
    return_polygon : bool, default False
        Also return the polygon vertices.
    linewidth : float, default 1.25
        The outline width.

    Returns
    -------
    ax : cartopy.mpl.geoaxes.GeoAxes
        The axes.
    polygon_lat : numpy.ndarray
        Latitudes of the polygon vertices; only with `return_polygon`.
    polygon_lon : numpy.ndarray
        Longitudes of the polygon vertices; only with `return_polygon`.
    polygon_defined : bool
        Whether the polygon exists for this grid; only with `return_polygon`.
    """
    from hcwf.onset.regions import polygon_boundary

    polygon_defined = False

    if polygon:
        polygon1_lat, polygon1_lon = polygon_boundary(da)

        if len(polygon1_lat) > 0 and len(polygon1_lon) > 0:
            polygon_defined = True

    # Add CMZ polygon only if defined
    if polygon_defined:

        ax = polygon_outline(ax, polygon1_lon, polygon1_lat, linewidth=linewidth)

    if return_polygon:
        return ax, polygon1_lat, polygon1_lon, polygon_defined
    else:
        return ax



def get_india_outline(shpfile_path):
    """Get the region outline coordinates from a shapefile.

    Parameters
    ----------
    shpfile_path : str or pathlib.Path
        The shapefile, read with `geopandas.read_file`.

    Returns
    -------
    list of tuple of (list of float, list of float)
        The ``(lons, lats)`` of the exterior ring of every polygon (the parts of multi-polygons included).
    """
    import geopandas as gpd
    # Update this path to your India shapefile
    india_gdf = gpd.read_file(shpfile_path)

    boundaries = []
    for geom in india_gdf.geometry:
        if hasattr(geom, 'exterior'):
            coords = list(geom.exterior.coords)
            lon_coords = [coord[0] for coord in coords]
            lat_coords = [coord[1] for coord in coords]
            boundaries.append((lon_coords, lat_coords))
        elif hasattr(geom, 'geoms'):
            for sub_geom in geom.geoms:
                if hasattr(sub_geom, 'exterior'):
                    coords = list(sub_geom.exterior.coords)
                    lon_coords = [coord[0] for coord in coords]
                    lat_coords = [coord[1] for coord in coords]
                    boundaries.append((lon_coords, lat_coords))
    return boundaries


def shp_outline(ax, region='Ethiopia', resolution='10m'):
    """Add the country boundary of a region to a plot.

    On a Cartopy axes the geometry is added with coastlines and borders; on a plain axes its exterior is plotted
    as lines. A region that is not found is logged as a warning and nothing is drawn.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axes to draw on.
    region : str, default "Ethiopia"
        The region (country) name, looked up with `hcwf.spatial.masks.get_shp`.
    resolution : str, default "10m"
        The Natural Earth resolution.

    Returns
    -------
    matplotlib.axes.Axes
        `ax`.
    """
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature

    # get region boundary
    from hcwf.spatial.masks import get_shp

    region_geom = get_shp(region=region, resolution=resolution)

    if region_geom is None:
        log.warning("specified region is not in cartopy.io.shapereader")
        return ax

    if hasattr(ax, "add_geometries"):
        # Cartopy GeoAxes
        ax.add_geometries(
            [region_geom],
            crs=ccrs.PlateCarree(),
            facecolor="none",
            edgecolor="black",
            linewidth=1.5,
            zorder=10
        )
    else:
        # Regular matplotlib Axes — handle both Polygon and MultiPolygon
        geoms = [region_geom] if region_geom.geom_type == 'Polygon' else list(region_geom.geoms)
        for geom in geoms:
            x, y = geom.exterior.xy
            ax.plot(x, y, color="black", linewidth=1.5, zorder=10)

    # Optional: background
    if hasattr(ax, "add_feature"):
        ax.add_feature(cfeature.COASTLINE)
        ax.add_feature(cfeature.BORDERS, linestyle=":")

    return ax
