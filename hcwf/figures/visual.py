"""Plotting helpers: monthly onset colormap, map basemaps and boxes, and portrait/heatmap plots.

The shared building blocks of the figure modules of `hcwf.figures`: the season colormap (`cbar_season`), the region
basemap and outlines (`set_basemap`, `shp_outline`, `shapefile_outline`, `draw_region_outline`, `polygon_outline`, `add_polygon`,
`box_boundary`) and the portrait plot (`portrait_plot` with `prepare_data`, `heatmap`, `annotate_heatmap`),
a simplified version of the PCMDI Metrics Package (PMP) portrait plot.
"""
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
        The norm mapping day of year onto ``cmap_jjas``.
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
        The run's `~hcwf.config.model.FigureSpec`: its ``region`` gives the extent (and the outline), and ``shpfile_dir`` (a
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

    # Add the region outline
    draw_region_outline(ax, shapefile_outline(shpfile_dir) if shpfile_dir else None, region, 1.5)

    # Set the map extent to the region's box
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
        ``ax`` (the rectangle is added to it).
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
    debug=False,
):
    """Draw a portrait plot: a heatmap of a 2D array with labelled rows and columns.

    Only the heatmap style (a single 2D array) is drawn; stacked arrays (the triangle styles of the PMP
    ``portrait_plot()``, which this simplifies) are not supported. Raises ValueError when the inputs are inconsistent.

    Parameters
    ----------
    data : numpy.ndarray or list of numpy.ndarray
        A 2D array, or a list holding one 2D array, with shape (row, column).
    xaxis_labels : list of str
        The column labels; the length equals the number of columns, or is 0 to turn off the labels.
    yaxis_labels : list of str
        The row labels; the length equals the number of rows, or is 0 to turn off the labels.
    fig : matplotlib.figure.Figure, optional
        The figure to plot on. When None a new figure is created.
    ax : matplotlib.axes.Axes, optional
        The axes to plot on. When None a new subplot of ``fig`` is created.
    annotate : bool, default False
        Add the value as text in each cell.
    annotate_data : numpy.ndarray, optional
        The values written in the cells. None uses ``data``.
    annotate_textcolors : tuple of (str, str), default ("black", "white")
        The text colours below and above ``annotate_textcolors_threshold``.
    annotate_textcolors_threshold : tuple of (float, float), optional
        Passed to `annotate_heatmap` as its ``threshold``: the text colour switches outside it.
    annotate_fontsize : float, default 15
        The font size of the annotations (divided by half the number of columns).
    annotate_format : str, default "{x:.2f}"
        The format of the annotated values.
    figsize : tuple of (float, float), default (12, 10)
        The figure size in inches, when a figure is created.
    vrange : tuple of (float, float), optional
        The range of the colour scale. None uses the minimum and maximum of ``data``.
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
        The colorbar; returned (as a fourth value) only when ``colorbar_off`` is false.

    Raises
    ------
    ValueError
        When ``data`` is stacked, or the labels or ``annotate_data`` do not match its shape.
    """
    # ----------------
    # Prepare plotting
    # ----------------
    data, num_divide = prepare_data(data, xaxis_labels, yaxis_labels, debug=debug)

    if num_divide != 1:
        raise ValueError("portrait_plot draws a single 2D array; stacked arrays are not supported")

    if annotate:
        if annotate_data is None:
            annotate_data = data
        else:
            annotate_data, _ = prepare_data(annotate_data, xaxis_labels, yaxis_labels, debug=debug)
            if annotate_data.shape != data.shape:
                raise ValueError("annotate_data does not have the same shape as data")

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

    # Heatmap-style portrait plot
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
    """Convert ``data`` to a numpy array and check its shape against the labels.

    Raises ValueError when the data are not a 2D or 3D array, a list that is not made of 2D arrays, or when the
    labels do not match the shape (an empty label list matches anything).

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

    Raises
    ------
    ValueError
        When the data or the labels are not as described above.
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
                raise ValueError("Element of given list is not in np.ndarray type")
        else:  # list has more than 1 arrays as elements
            data = np.stack(data)
            num_divide = len(data)

    # Now, data is expected to be be a numpy array (whether given or converted from list)
    if debug:
        log.debug("%s %s", "data.shape:", data.shape)

    if data.shape[-1] != len(xaxis_labels) and len(xaxis_labels) > 0:
        raise ValueError("Number of elements in xaxis_label mismatchs to the data")

    if data.shape[-2] != len(yaxis_labels) and len(yaxis_labels) > 0:
        raise ValueError("Number of elements in yaxis_label mismatchs to the data")

    if isinstance(data, np.ndarray):
        if len(data.shape) == 2:
            num_divide = 1
        elif len(data.shape) == 3:
            num_divide = data.shape[0]
        else:
            log.debug("%s %s", "data.shape:", data.shape)
            raise ValueError("data.shape is not right")
    else:
        raise ValueError("Converted or given data is not in np.ndarray type")

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
    norm from the 10 % and 90 % quantiles of ``data``); with a tuple ``threshold`` it is picked from ``textcolors``
    instead. The font size is ``fontsize * 2 / number of columns`` (``fontsize`` from ``textkw``, default the
    matplotlib ``font.size``).

    Parameters
    ----------
    im : matplotlib.collections.QuadMesh
        The drawn heatmap to label. Its array is used when ``data`` is not a list or array.
    ax : matplotlib.axes.Axes
        The axes of the heatmap.
    data : numpy.ndarray, shape (row, column), optional
        The data the image is coloured with.
    cmap : str or matplotlib.colors.Colormap, optional
        The colormap of the image.
    annotate_data : numpy.ndarray, shape (row, column), optional
        The values written in the cells. None uses ``data``.
    valfmt : str or matplotlib.ticker.Formatter, default "{x:.2f}"
        The format of the annotations: a `str.format` string with ``x``, or a formatter.
    textcolors : tuple of (str, str), default ("black", "white")
        The colours of values inside and outside a tuple ``threshold`` (``(low, high)``).
    threshold : tuple of (float, float), optional
        The cells outside ``[min, max]`` get the second colour. None applies the contrast rule.
    **textkw : dict
        Passed on to `matplotlib.axes.Axes.text` for every cell; ``fontsize`` is scaled as above.

    Returns
    -------
    matplotlib.axes.Axes
        ``ax``, with a text per cell.
    """
    if not isinstance(data, (list, np.ndarray)):
        data = im.get_array().reshape(im._meshHeight, im._meshWidth)

    if annotate_data is None:
        annotate_data = data

    # Set default alignment to center, but allow it to be
    # overwritten by textkw.
    kw = dict(horizontalalignment="center", verticalalignment="center")
    kw.update(textkw)

    # adjust annote fontsize according to # of heat blocks
    x_num = data.shape[1]
    kw['fontsize'] = kw.get('fontsize', plt.rcParams['font.size']) * 2 / x_num

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
            if threshold is not None:
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
        The norm mapping ``value`` to the colormap.
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
        ``ax``.
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
    """Draw the polygon boundary for a field on ``ax`` when ``polygon`` is truthy and a boundary is defined.

    Parameters
    ----------
    ax : cartopy.mpl.geoaxes.GeoAxes
        The axes to draw on.
    da : xarray.DataArray
        The field; the boundary depends on its grid (see `hcwf.onset.regions.polygon_boundary`).
    polygon : bool
        Draw the boundary; with a false value nothing is drawn (and ``return_polygon`` must be false).
    return_polygon : bool, default False
        Also return the polygon vertices.
    linewidth : float, default 1.25
        The outline width.

    Returns
    -------
    ax : cartopy.mpl.geoaxes.GeoAxes
        The axes.
    polygon_lat : numpy.ndarray
        Latitudes of the polygon vertices; only with ``return_polygon``.
    polygon_lon : numpy.ndarray
        Longitudes of the polygon vertices; only with ``return_polygon``.
    polygon_defined : bool
        Whether the polygon exists for this grid; only with ``return_polygon``.
    """
    from hcwf.onset.regions import polygon_boundary

    polygon_defined = False
    polygon1_lat = polygon1_lon = np.array([])

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



def shapefile_outline(shpfile_path):
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
    gdf = gpd.read_file(shpfile_path)

    boundaries = []
    for geom in gdf.geometry:
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


def draw_region_outline(ax, boundaries, region, linewidth):
    """Draw the region outline on ``ax``: the given boundary lines, or else the region's country outline.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axes to draw on.
    boundaries : list of tuple or None
        The ``(lon, lat)`` boundary lines from `shapefile_outline`; None draws the outline of ``region`` with
        `shp_outline` instead.
    region : str
        The region name, for `shp_outline`.
    linewidth : float
        The width of the boundary lines.
    """
    if boundaries is None:
        shp_outline(ax, region=region)
    else:
        for lon, lat in boundaries:
            ax.plot(lon, lat, color='black', linewidth=linewidth)


def shp_outline(ax, region, resolution='10m'):
    """Add the country boundary of a region to a plot.

    On a Cartopy axes the geometry is added with coastlines and borders; on a plain axes its exterior is plotted
    as lines. A region that is not found is logged as a warning and nothing is drawn.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axes to draw on.
    region : str
        The region (country) name, looked up with `hcwf.spatial.masks.get_shp`.
    resolution : str, default "10m"
        The Natural Earth resolution.

    Returns
    -------
    matplotlib.axes.Axes
        ``ax``.
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
