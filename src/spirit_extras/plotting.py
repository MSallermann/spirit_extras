import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
import matplotlib as mpl
import numpy as np


class Paper_Plot:
    # one cm in inches
    cm = 1 / 2.54

    # Annotations
    offset_u = 1.5 * np.array([0, 10])
    offset_r = 1.5 * np.array([10, 0])
    offset_l = -1.5 * offset_r
    offset_d = -1.5 * offset_u
    offset_ur = (offset_r + offset_u) / np.sqrt(2)
    offset_ul = (offset_l + offset_u) / np.sqrt(2)
    offset_dr = (offset_r + offset_d) / np.sqrt(2)
    offset_dl = (offset_l + offset_d) / np.sqrt(2)

    offset_dict = {
        "u": offset_u,
        "r": offset_r,
        "l": offset_l,
        "d": offset_d,
        "ur": offset_ur,
        "ul": offset_ul,
        "dr": offset_dr,
        "dl": offset_dl,
    }

    default_rcParams = {
        "font.size": 8,
        "font.family": "serif",
        "mathtext.fontset": "dejavuserif",
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "axes.labelsize": 8,
    }

    # __slots__ = ["annotate_letter", "width", "height", "ncols", "nrows"]

    def __init__(self, width, height=None, nrows=1, ncols=1, rcParams=None) -> None:

        self._ncols = ncols
        self._nrows = nrows
        self.width = width

        if height is None:
            height = width

        self.height = height

        if rcParams is None:
            rcParams = self.default_rcParams

        mpl.rcParams.update(rcParams)

        self.annotate_letter = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        # self.annotate_letter = "ABCDEFGHIJKLMNOPQRSTUVWXYZ".lower()
        # self.annotate_letter = ["I", "II", "III", "IV", "V", "VI", "VII", "VIII", "IX", "X"]
        # self.annotate_letter = ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10"]

        self._horizontal_margins = [0.1, 0.1]
        self._vertical_margins = [0.1, 0.1]
        self._wspace = 0.1
        self._hspace = 0.1

        self._width_ratios = None
        self._height_ratios = None

        self._fig = None
        self._gs = None

        self.annotation_dict = {}

    def info_string(self):
        res = "Paper_Plot\n"
        res += f"\t width  = {self.width:.3f} inch ({self.width / self.cm:.3f} cm)\n"
        res += f"\t height = {self.height:.3f} inch ({self.height / self.cm:.3f} cm)\n"
        res += f"\t ncols  = {self.ncols}\n"
        res += f"\t nrows  = {self.nrows}\n"
        res += f"\t wspace = {self.wspace}\n"
        res += f"\t hspace = {self.hspace}\n"
        res += f"\t horizontal_margins = {self.horizontal_margins}\n"
        res += f"\t vertical_margins = {self.vertical_margins}\n"
        res += f"\t width_ratios = {self.width_ratios}\n"
        res += f"\t height_ratios = {self.height_ratios}\n"
        return res

    @property
    def ncols(self):
        return self._ncols

    @ncols.setter
    def ncols(self, value):
        if value != self._ncols and self._width_ratios:
            print("WARNING: changing ncols resets width_ratios")
            self._width_ratios = None
        self._ncols = value

    @property
    def nrows(self):
        return self._nrows

    @nrows.setter
    def nrows(self, value):
        if value != self._nrows and self._height_ratios:
            print("WARNING: changing ncols resets height_ratios")
            self._height_ratios = None
        self._nrows = value

    @property
    def font_size(self):
        return mpl.rcParams["font.size"]

    @font_size.setter
    def font_size(self, value):
        mpl.rcParams["font.size"] = value

    @property
    def axes_labelsize(self):
        return mpl.rcParams["axes.labelsize"]

    @axes_labelsize.setter
    def axes_labelsize(self, value):
        mpl.rcParams["axes.labelsize"] = value

    @property
    def xtick_labelsize(self):
        return mpl.rcParams["xtick.labelsize"]

    @xtick_labelsize.setter
    def axes_labelsize(self, value):
        mpl.rcParams["xtick.labelsize"] = value

    @property
    def ytick_labelsize(self):
        return mpl.rcParams["ytick.labelsize"]

    @ytick_labelsize.setter
    def axes_labelsize(self, value):
        mpl.rcParams["ytick.labelsize"] = value

    @property
    def width_ratios(self):
        return self._width_ratios

    @width_ratios.setter
    def width_ratios(self, value):
        if value is None:
            self._width_ratios = None
            return

        if len(value) == self.ncols:
            self._width_ratios = value
        else:
            raise Exception(f"Length of width_ratios has to match ncols {self.ncols}")

    @property
    def height_ratios(self):
        return self._height_ratios

    @height_ratios.setter
    def height_ratios(self, value):
        if value is None:
            self._height_ratios = None
            return

        if len(value) == self.nrows:
            self._height_ratios = value
        else:
            raise Exception(f"Length of height_ratios has to match nrows {self.nrows}")

    @property
    def wspace(self):
        return self._wspace

    @wspace.setter
    def wspace(self, value):
        self._wspace = value

    @property
    def hspace(self):
        return self._hspace

    @hspace.setter
    def hspace(self, value):
        self._hspace = value

    @property
    def horizontal_margins(self):
        return self._horizontal_margins.copy()

    @horizontal_margins.setter
    def horizontal_margins(self, value):
        if np.asarray(value).shape == (2,):
            self._horizontal_margins = value
        else:
            raise Exception(
                "horizontal_margins has to have shape (2,) but you specified {}".format(
                    value
                )
            )

    @property
    def vertical_margins(self):
        return self._vertical_margins.copy()

    @vertical_margins.setter
    def vertical_margins(self, value):
        if np.asarray(value).shape == (2,):
            self._vertical_margins = value
        else:
            raise Exception(
                "vertical_margins has to have shape (2,) but you specified {}".format(
                    value
                )
            )

    def height_from_aspect_ratio(self, aspect_ratio):
        """The width of the figure (including all margins) is fixed. The aspect ratio is the aspect ratio of all content, excluding hspace and wspace"""

        # Deal with width
        rel_margin_w = sum(self.horizontal_margins)
        rel_width_minus_margins = 1.0 - rel_margin_w

        rel_average_subplot_width = rel_width_minus_margins / (
            self.ncols + self.wspace * (self.ncols - 1)
        )

        rel_total_wspace_between_subplots = (
            rel_average_subplot_width * self.wspace * (self.ncols - 1)
        )

        width_prefactor = rel_width_minus_margins - rel_total_wspace_between_subplots

        rel_margin_h = sum(self.vertical_margins)
        rel_height_minus_margins = 1 - rel_margin_h

        rel_average_subplot_height = rel_height_minus_margins / (
            self.nrows + self.hspace * (self.nrows - 1)
        )

        rel_total_hspace_between_subplots = (
            rel_average_subplot_height * self.hspace * (self.nrows - 1)
        )
        height_prefactor = rel_height_minus_margins - rel_total_hspace_between_subplots

        # aspect_ratio = self.width/self.height * width_prefactor / height_prefactor
        self.height = self.width / aspect_ratio * width_prefactor / height_prefactor

    def apply_absolute_margins(
        self,
        aspect_ratio,
        abs_hspace,
        abs_wspace,
        abs_vertical_margins,
        abs_horizontal_margins,
    ):
        abs_horizontal_margins = np.array(abs_horizontal_margins, dtype=float)
        abs_vertical_margins = np.array(abs_vertical_margins, dtype=float)

        abs_margin_w = np.sum(abs_horizontal_margins)
        abs_margin_h = np.sum(abs_vertical_margins)

        abs_content_width = self.width - abs_wspace * (self.ncols - 1) - abs_margin_w
        if abs_content_width <= 0:
            raise Exception(
                "horizontal margins and wspace are too big. There is no space left for the content."
            )

        abs_content_height = abs_content_width / aspect_ratio

        self.hspace = abs_hspace / abs_content_height * self.nrows

        self.wspace = abs_wspace / abs_content_width * self.ncols
        self.height = abs_content_height + abs_margin_h + abs_hspace * (self.nrows - 1)

        self.horizontal_margins = abs_horizontal_margins / self.width
        self.vertical_margins = abs_vertical_margins / self.height

    def fig(self):
        self._fig = plt.figure(figsize=(self.width, self.height))
        return self._fig

    def gs(self):
        self._gs = GridSpec(
            figure=self._fig,
            nrows=self.nrows,
            ncols=self.ncols,
            left=self.horizontal_margins[0],
            bottom=self.vertical_margins[0],
            right=1 - self.horizontal_margins[1],
            top=1 - self.vertical_margins[1],
            hspace=self.hspace,
            wspace=self.wspace,
            width_ratios=self.width_ratios,
            height_ratios=self.height_ratios,
        )
        return self._gs

    def annotate(self, ax, text, pos=[0, 0.98], fontsize=8):
        ax.text(
            *pos,
            text,
            fontsize=fontsize,
            horizontalalignment="left",
            verticalalignment="top",
            transform=ax.transAxes,
        )

    def crop_to_content(
        self, image, background_color=None, replace_background_color=None
    ):
        N_CHANNELS = image.shape[-1]  # Number of channels in the picture
        image_shape = image.shape

        # If no background color is specified, we take the most frequent color among the corners
        if background_color is None:
            corner_colors = [image[0, 0], image[0, -1], image[-1, 0], image[-1, -1]]

            # As soon as we find the first repeated color we can stop, since a count of two will always be at least 50%
            # TODO: these loops look stupid but they work
            for cc in corner_colors:
                for cc2 in corner_colors:
                    if np.all(cc == cc2):
                        background_color = cc
                        break
                if np.all(cc == cc2):
                    break

            background_color = corner_colors[0]

        # I think I know why this works
        indices = np.argwhere(np.any(image[:, :] != background_color, axis=2))

        lower_height = np.min(indices[:, 0])
        upper_height = np.max(indices[:, 0])
        lower_width = np.min(indices[:, 1])
        upper_width = np.max(indices[:, 1])

        if not replace_background_color is None:
            N_CHANNELS_BG = len(replace_background_color)

            # If the background color has more channels than the picture, we need to introduce the alpha channel
            if N_CHANNELS_BG > N_CHANNELS and replace_background_color[-1] != 1.0:
                image_copy = np.ones(shape=(image_shape[0], image_shape[1], 4))
                image_copy[:, :, :3] = image
                image = image_copy
                background_color = [
                    *background_color,
                    1.0,
                ]  # Extend the background color with the alpha channel
            elif N_CHANNELS_BG < N_CHANNELS:
                replace_background_color = [*replace_background_color, 1.0]

            indices = np.argwhere(np.all(image[:, :] == background_color, axis=2))
            image[indices[:, 0], indices[:, 1], :] = replace_background_color

        return image[lower_height:upper_height, lower_width:upper_width, :]

    def crop(self, image, width, height=None):
        if height is None:
            height = image.shape[0]
        o = [int(image.shape[0] / 2), int(image.shape[1] / 2)]
        lower_height = o[0] - int(height / 2)
        lower_width = o[1] - int(width / 2)
        upper_width = lower_width + width
        upper_height = lower_height + height
        return image[lower_height:upper_height, lower_width:upper_width, :]

    def image_to_ax(self, ax, image):
        import os

        if os.path.exists(image):
            image = plt.imread(image)
        ax.imshow(image)
        ax.tick_params(
            axis="both", which="both", bottom=0, left=0, labelbottom=0, labelleft=0
        )
        ax.set_facecolor([0, 0, 0, 0])
        for k, s in ax.spines.items():
            s.set_visible(False)

    def spine_axis(
        self,
        spec,
        color="black",
        which=["left", "right", "top", "bottom"],
        zorder=2,
        label="spine",
    ):
        try:
            a = self._fig.add_axes(
                spec.get_position(self._fig), zorder=zorder, label=label
            )
        except:
            a = self._fig.add_axes(spec, zorder=zorder, label=label)

        a.set_facecolor([0, 0, 0, 0])
        a.tick_params(
            axis="both", which="both", bottom=0, left=0, labelbottom=0, labelleft=0
        )
        for k in ["left", "right", "top", "bottom"]:
            s = a.spines[k]
            if k in which:
                s.set_visible(True)
                s.set_color(color)
            else:
                s.set_visible(False)
        return a

    def row(self, row_idx, sl=slice(None, None, None), gs=None):
        if gs is None:
            gs = self._gs

        col_indices = range(gs.ncols)[sl]
        return [self._fig.add_subplot(gs[row_idx, col_idx]) for col_idx in col_indices]

    def col(self, col_idx, sl=slice(None, None, None), gs=None):
        if gs is None:
            gs = self._gs

        row_indices = list(range(gs.nrows)[sl])
        return [self._fig.add_subplot(gs[row_idx, col_idx]) for row_idx in row_indices]

    def xy_text_auto(self, ax, xy, deriv, scale=15):
        trans_deriv = ax.transData.transform(
            [[0, 0], [1, deriv]]
        )  # transform from data coordinates to display coordinates
        display_deriv = (trans_deriv[1, 1] - trans_deriv[0, 1]) / (
            trans_deriv[1, 0] - trans_deriv[0, 0]
        )

        # display_deriv = trans_deriv[1]/trans_deriv[0] # transform to display derivative
        direction = np.array([display_deriv, -1])
        direction = direction / np.linalg.norm(direction)

        # Where do we hit the upper ylim in display coords
        lim_lower = ax.transData.transform([ax.get_xlim()[0], ax.get_ylim()[0]])
        lim_upper = ax.transData.transform([ax.get_xlim()[1], ax.get_ylim()[1]])
        xy = ax.transData.transform(xy)

        intersect_upper = (lim_upper - xy) / direction
        dist_upper = np.min(np.abs(intersect_upper))
        sign_upper = np.sign(intersect_upper[0])

        intersect_lower = (lim_lower - xy) / direction
        dist_lower = np.min(np.abs(intersect_lower))
        sign_lower = np.sign(intersect_lower[0])

        if dist_upper > dist_lower:
            return scale * sign_upper * direction
        else:
            return scale * sign_lower * direction

    def annotate_graph(self, ax, xy, xy_text, text=None, key="key1", offset_scale=1):

        if not key is None:
            if not key in self.annotation_dict:
                self.annotation_dict[key] = {
                    "annotate_increment": 0,
                    "annotation_list": [],
                }
        elif text is None:
            raise Exception("Need to specify text if key is None")

        arrowprops = dict(arrowstyle="-")

        if text is None:
            text = self.annotate_letter[self.annotation_dict[key]["annotate_increment"]]
            self.annotation_dict[key]["annotate_increment"] += 1

        if type(xy_text) is str:
            xy_text = Paper_Plot.offset_dict[xy_text.lower()]

        if not key is None:
            self.annotation_dict[key]["annotation_list"].append([xy, text])

        ax.annotate(
            text,
            xy,
            xy_text * offset_scale,
            arrowprops=arrowprops,
            verticalalignment="center",
            horizontalalignment="center",
            textcoords="offset points",
        )


def colorbar(mappable, *args, **kwargs):
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    import matplotlib.pyplot as plt

    last_axes = plt.gca()
    ax = mappable.axes
    fig = ax.figure
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = fig.colorbar(mappable, cax=cax, *args, **kwargs)
    plt.sca(last_axes)
    return cbar


def plot_basis_cell(p_state, ax=None):
    from spirit import geometry, system

    # Read out the information we need
    n_cell_atoms = geometry.get_n_cell_atoms(p_state)
    n_cells = geometry.get_n_cells(p_state)
    positions = geometry.get_positions(p_state)

    def idx(i, a, b, c):
        return i + n_cell_atoms * (a + n_cells[0] * (b + n_cells[1] * c))

    # Draw the outlines of the basis cell
    lattice_points = np.array(
        [
            positions[idx(0, 0, 0, 0)],
            positions[idx(0, 1, 0, 0)],
            positions[idx(0, 1, 1, 0)],
            positions[idx(0, 0, 1, 0)],
        ]
    )

    patch = [Polygon(lattice_points[:, :2])]
    pc = PatchCollection(
        patch, linestyle="--", facecolor=[0, 0, 0, 0], edgecolor=[0, 0, 0, 1]
    )
    ax.add_collection(pc)
    ax.scatter(
        positions[:n_cell_atoms, 0], positions[:n_cell_atoms, 1], color="red", s=100
    )
    ax.set_xlabel(r"$x\;\;[\AA]$")
    ax.set_ylabel(r"$y\;\;[\AA]$")


def set_kwarg_if_not_there(kwarg_dict, key, value):
    if not key in kwarg_dict:
        kwarg_dict[key] = value


def plot_spins_2d(
    spin_system, ax, col_xy=[0, 1], x=None, y=None, u=None, v=None, c=None, **kwargs
):

    set_kwarg_if_not_there(kwargs, "lw", 0.5)
    set_kwarg_if_not_there(kwargs, "units", "xy")
    set_kwarg_if_not_there(kwargs, "edgecolor", "black")
    set_kwarg_if_not_there(kwargs, "scale", 2)
    set_kwarg_if_not_there(kwargs, "width", 0.13)
    set_kwarg_if_not_there(kwargs, "pivot", "mid")
    set_kwarg_if_not_there(kwargs, "cmap", "seismic")
    set_kwarg_if_not_there(kwargs, "alpha", 1)
    set_kwarg_if_not_there(kwargs, "clim", (-1, 1))

    if not (x and y):
        x = spin_system.positions[:, col_xy[0]]
        y = spin_system.positions[:, col_xy[1]]

    if not (u and v and c):
        u = spin_system.spins[:, col_xy[0]]
        v = spin_system.spins[:, col_xy[1]]
        color_column = [i for i in [0, 1, 2] if i not in col_xy]
        c = spin_system.spins[:, color_column]

    ax.set_aspect("equal")

    cb = ax.quiver(x, y, u, v, c, **kwargs)
    ax.scatter(x, y, marker=".", color="black", s=1)
    ax.set_xlabel(r"$x~[\AA]$")
    ax.set_ylabel(r"$y~[\AA]$")
    colorbar(cb, label=r"$\mathbf{m}_z$")


def plot_energy_path(
    energy_path,
    ax,
    normalize_reaction_coordinate=False,
    kwargs_discrete={},
    kwargs_interpolated={},
    plot_interpolated=True,
):

    set_kwarg_if_not_there(kwargs_discrete, "markeredgecolor", "black")
    set_kwarg_if_not_there(kwargs_discrete, "marker", "o")
    set_kwarg_if_not_there(kwargs_discrete, "ls", "None")
    set_kwarg_if_not_there(kwargs_discrete, "color", "C0")

    set_kwarg_if_not_there(kwargs_interpolated, "lw", 1.75)
    set_kwarg_if_not_there(kwargs_interpolated, "marker", "None")
    set_kwarg_if_not_there(kwargs_interpolated, "ls", "-")
    set_kwarg_if_not_there(kwargs_interpolated, "color", kwargs_discrete["color"])

    E0 = energy_path.total_energy[-1]

    if len(energy_path.interpolated_reaction_coordinate) > 0 and plot_interpolated:
        Rx_int = np.array(energy_path.interpolated_reaction_coordinate)
        if normalize_reaction_coordinate:
            Rx_int = Rx_int / Rx_int[-1]
        ax.plot(
            Rx_int,
            np.asarray(energy_path.interpolated_total_energy) - E0,
            **kwargs_interpolated,
        )

    Rx = np.array(energy_path.reaction_coordinate)
    if normalize_reaction_coordinate:
        Rx = Rx / Rx[-1]
    ax.plot(Rx, np.asarray(energy_path.total_energy) - E0, **kwargs_discrete)

    ax.set_xlabel("reaction coordinate [arb. units]")
    ax.set_ylabel(r"$\Delta E$ [meV]")


def gradient_line(ax, x, y, c, lw=2.0, cmap="viridis", c_norm=None, n_inter=10):
    """Plots a line with a color gradient
    Usage example:
        x = np.linspace(0, 1, 30)
        y = np.sin(x*5)
        c = np.cos(x)
        gradient_line(ax = plt.gca(), x=x, y=y, c=c)
    """
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection
    from matplotlib.colors import Normalize

    try:
        x_dense = np.unique(
            [np.linspace(x[i], x[i + 1], n_inter) for i in range(len(x) - 1)]
        )
        y_dense = np.interp(x_dense, x, y)
        c_dense = np.interp(x_dense, x, c)

        segments = np.zeros(shape=(len(x_dense) - 1, 4))

        segments[:, 0] = x_dense[:-1]
        segments[:, 1] = y_dense[:-1]
        segments[:, 2] = x_dense[1:]
        segments[:, 3] = y_dense[1:]

        if c_norm is None:
            norm = plt.Normalize(c_dense.min(), c_dense.max())
        else:
            norm = plt.Normalize(*c_norm)

        segments = segments.reshape(len(x_dense) - 1, 2, 2)

        lc = LineCollection(segments, cmap=cmap, norm=norm)

        lc.set_array(c_dense)
        lc.set_linewidth(lw)

        ax.add_collection(lc)
        # ax.autoscale()
    except:
        pass


def get_rgba_colors_red_blue(spins, opacity=1.0, cardinal=np.array([0, 0, 1])):
    cardinal = np.asarray(cardinal)

    def mix(x, y, a):
        return x * (1.0 - a) + y * a

    red = np.array([1, 0, 0, opacity])
    blue = np.array([0, 0, 1, opacity])

    rgba = []

    for spin in spins:
        projection_z = cardinal.dot(spin)
        _rgba = mix(red, blue, (projection_z + 1.0) / 2.0)
        rgba.append([_rgba])

    return rgba


def get_rgba_colors_red_green_blue(spins, opacity=1.0, cardinal=np.array([0, 0, 1])):
    cardinal = np.asarray(cardinal)

    def mix(x, y, a):
        return x * (1.0 - a) + y * a

    red = np.array([1, 0, 0, opacity])
    green = np.array([0, 1, 0, opacity])
    blue = np.array([0, 0, 1, opacity])

    rgba = []

    for spin in spins:
        projection_z = cardinal.dot(spin)

        if projection_z < 0:
            _rgba = mix(green, red, -projection_z)

        if projection_z >= 0:
            _rgba = mix(green, blue, projection_z)

        rgba.append([_rgba])

    return rgba


def get_rgba_colors(
    spins,
    opacity=1.0,
    cardinal_a=np.array([1, 0, 0]),
    cardinal_b=np.array([0, 1, 0]),
    cardinal_c=np.array([0, 0, 1]),
):
    """Returns a colormap in the matplotlib format (an Nx4 array)"""
    import math

    # Annoying OpenGl functions

    def atan2(y, x):
        if x == 0.0:
            return np.sign(y) * np.pi / 2.0
        else:
            return np.arctan2(y, x)

    def fract(x):
        return x - math.floor(x)

    def mix(x, y, a):
        return x * (1.0 - a) + y * a

    def clamp(x, minVal, maxVal):
        return min(max(x, minVal), maxVal)

    def hsv2rgb(c):
        K = [1.0, 2.0 / 3.0, 1.0 / 3.0, 3.0]

        px = abs(fract(c[0] + K[0]) * 6.0 - K[3])
        py = abs(fract(c[0] + K[1]) * 6.0 - K[3])
        pz = abs(fract(c[0] + K[2]) * 6.0 - K[3])
        resx = c[2] * mix(K[0], clamp(px - K[0], 0.0, 1.0), c[1])
        resy = c[2] * mix(K[0], clamp(py - K[0], 0.0, 1.0), c[1])
        resz = c[2] * mix(K[0], clamp(pz - K[0], 0.0, 1.0), c[1])

        return [resx, resy, resz]

    rgba = []

    for spin in spins:
        phi_offset = 0
        projection_x = cardinal_a.dot(spin)
        projection_y = cardinal_b.dot(spin)
        projection_z = cardinal_c.dot(spin)
        hue = atan2(projection_x, projection_y) / (2 * np.pi)

        saturation = projection_z

        if saturation > 0.0:
            saturation = 1.0 - saturation
            value = 1.0
        else:
            value = 1.0 + saturation
            saturation = 1.0

        rgba.append([*hsv2rgb([hue, saturation, value]), opacity])

    return rgba
