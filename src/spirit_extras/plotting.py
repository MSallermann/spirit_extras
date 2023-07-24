import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
import matplotlib as mpl
import numpy as np
from matplotlib.patches import FancyBboxPatch
from PIL import Image


class Paper_Plot:
    """A class for laying out plots in a grid. (Plus some utilities)"""

    # one cm in inches
    cm = 1.0 / 2.54

    # one inch in inches
    inch = 1.0

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

    def __init__(self, width, height=None, nrows=1, ncols=1, rcParams=None) -> None:

        self._ncols = ncols
        self._nrows = nrows
        self.width = width
        self.height = height

        if rcParams is None:
            rcParams = self.default_rcParams

        mpl.rcParams.update(rcParams)

        self.annotate_letter = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

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
        """Returns a string with information about the plot."""

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
        """Number of columns."""
        return self._ncols

    @ncols.setter
    def ncols(self, value):
        if value != self._ncols and self._width_ratios:
            print("WARNING: changing ncols resets width_ratios")
            self._width_ratios = None
        self._ncols = value

    @property
    def nrows(self):
        """Number of rows."""
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
        """Relative column width ratios."""
        if value is None:
            self._width_ratios = None
            return

        if len(value) == self.ncols:
            self._width_ratios = value
        else:
            raise Exception(f"Length of width_ratios has to match ncols {self.ncols}")

    @property
    def height_ratios(self):
        """Relative row height ratios."""
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
        """Space between columns as a fraction of the average column width."""
        return self._wspace

    @wspace.setter
    def wspace(self, value):
        self._wspace = value

    @property
    def hspace(self):
        """Space between rows as a fraction of the average row height."""
        return self._hspace

    @hspace.setter
    def hspace(self, value):
        self._hspace = value

    @property
    def horizontal_margins(self):
        """Horizontal margins (left and right) as fraction of figure width"""
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
        """Horizontal margins (bottom and top) as fraction of figure height"""
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
        """The width of the figure (including all margins) is fixed. The aspect ratio is the aspect ratio of all content, excluding hspace, wspace and margins"""

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

        self.height = self.width / aspect_ratio * width_prefactor / height_prefactor

    def apply_absolute_margins(
        self,
        aspect_ratio=None,
        abs_hspace=0.5 * cm,
        abs_wspace=0.5 * cm,
        abs_vertical_margins=[0.15 * cm, 0.15 * cm],
        abs_horizontal_margins=[0.15 * cm, 0.15 * cm],
        abs_heights=None,
        abs_widths=None,
        abs_content_width=None,
        abs_content_height=None,
    ):
        """Set the layout of the figure in *absolute* units. All lengths are given in inches. Call this *before* fig() and gs().

        Args:
            aspect_ratio (float, optional): Aspect ratio of only the content of the entire figure, meaning all hspace, wspace and margins are discarded for the computation of the aspect ratio. Defaults to 1.62. Set to None to use different method to compute the conent height.
            abs_hspace (float): The space between each row of the grid. Defaults to 0.5*cm.
            abs_wspace (float): The space between each column of the grid. Defaults to 0.5*cm.
            abs_vertical_margins (list): Left and right margins. Defaults to [0.15 * cm, 0.15 * cm].
            abs_horizontal_margins (list): Bottom and top margins. Defaults to [0.15 * cm, 0.15 * cm].
            abs_heights (list, optional): Heights of the rows. Defaults to None. Negative values will be used as relative weights.
            abs_widths (list, optional): Widths of the columns. Defaults to None. Negative values will be used as relative weights.
            abs_content_width (float, optional): Absolute width of the content. !WARNING will recompute the total width of the figure, based on wspace and margins. It is usually better to keep the width fixed. Defaults to None.
            abs_content_height (float, optional): Absolute height of the content. This will only be used if aspect_ratio is None. !WARNING will recompute the total height of the figure.
        """

        # Convert arguments to np.arrayss
        abs_horizontal_margins = np.array(abs_horizontal_margins, dtype=float)
        abs_vertical_margins = np.array(abs_vertical_margins, dtype=float)

        assert abs_horizontal_margins.shape == (2,)
        assert abs_vertical_margins.shape == (2,)

        if not abs_widths is None:
            abs_widths = np.array(abs_widths, dtype=float)
            assert abs_widths.shape == (self.ncols,)

        if not abs_heights is None:
            abs_heights = np.array(abs_heights, dtype=float)
            assert abs_heights.shape == (self.nrows,)

        # Compute the absolute space, taken up by the margins
        abs_margin_w = np.sum(abs_horizontal_margins)
        abs_margin_h = np.sum(abs_vertical_margins)

        if abs_content_width is None:
            abs_content_width = (
                self.width - abs_wspace * (self.ncols - 1) - abs_margin_w
            )
        else:
            self.width = (
                abs_content_width + abs_wspace * (self.ncols - 1) + abs_margin_w
            )  # Re-compute the figure widths

        if abs_content_width <= 0.0:
            raise Exception(
                "Horizontal margins and wspace are too big. There is no space left for the content."
            )

        # Deal with absolute widths
        if not abs_widths is None:
            self.width_ratios = abs_widths / abs_content_width

            if (
                len(abs_widths[abs_widths < 0.0]) == 0
            ):  # If there are no relative widths left to distribute, we have no slack and check that the sum of absolute widths matches the conent width
                if not np.isclose(np.sum(abs_widths), abs_content_width):
                    raise Exception(
                        "The absolute widths do not match the expected width of the plot content. You should make at least one of them relative by specifying a negative number."
                    )

            # Compute the remaining width, to be distributed according to the relative weights
            remaining_width = abs_content_width - np.sum(abs_widths[abs_widths >= 0.0])
            if remaining_width < 0.0:
                raise Exception("Absolute widths are larger than total width")

            total_weight_of_relative_widths = np.sum(abs_widths[abs_widths < 0.0])

            # Divide the remaining width according to the relative width ratios
            for idx_w, w in enumerate(abs_widths):
                if w < 0.0:
                    self.width_ratios[idx_w] = (
                        w
                        / total_weight_of_relative_widths
                        * remaining_width
                        / abs_content_width
                    )

        # Check if overspecified
        height_specifiers = []
        if not aspect_ratio is None:
            height_specifiers.append("aspect_ratio")
        if not abs_content_height is None:
            height_specifiers.append("abs_content_height")
        if not self.height is None:
            height_specifiers.append("height")
        if not abs_heights is None:
            if np.all(abs_heights >= 0.0):
                height_specifiers.append(
                    "list of abs_heights without any relative weights"
                )

        if len(height_specifiers) > 1:
            raise Exception(
                f"The height of the plot is overspecified, because you have set {height_specifiers})"
            )

        # Decide on the content height
        if not aspect_ratio is None:
            abs_content_height = abs_content_width / aspect_ratio
        elif not abs_content_height is None:
            abs_content_height = abs_content_height
        elif not abs_heights is None:
            if np.all(
                abs_heights >= 0.0
            ):  # In some cases the abs content height can be computed from the given heights
                abs_content_height = np.sum(abs_heights)
        elif not self.height is None:
            abs_content_height = (
                self.height - abs_margin_h - abs_hspace * (self.nrows - 1)
            )
            if abs_content_height < 0:
                raise Exception(
                    "Absolute height for the content of figure is smaller than zero.\n"
                )
            print(
                "WARNING: Using the currently set height to compute the absolute content height."
            )
        else:
            raise Exception("Cannot determine height of the plot!")

        if not abs_heights is None:
            if (
                len(abs_heights[abs_heights < 0.0]) == 0
            ):  # If there are no relative heights to distribute, we have no slack and check that the sum of absolute widths matches the conent width
                if not np.isclose(np.sum(abs_heights), abs_content_height):
                    raise Exception(
                        "The sum of the absolute heights does not match the expected height of the plot *content*. You should make at least one of them relative by specifying a negative number."
                    )

            self.height_ratios = abs_heights / abs_content_height

            # Compute the remaining height to be distributed by relative weight
            remaining_height = abs_content_height - np.sum(
                abs_heights[abs_heights >= 0.0]
            )

            if remaining_height < 0.0:
                raise Exception("Absolute heights are larger than total height")

            total_weight_of_relative_heights = np.sum(abs_heights[abs_heights < 0])

            # divide the remaining height according to the relative height ratios
            for idx_h, h in enumerate(abs_heights):
                if h < 0.0:
                    self.height_ratios[idx_h] = (
                        h
                        / total_weight_of_relative_heights
                        * remaining_height
                        / abs_content_height
                    )

        # Compute the relative quantities that gridpsec needs
        self.hspace = abs_hspace / abs_content_height * self.nrows
        self.wspace = abs_wspace / abs_content_width * self.ncols
        self.height = abs_content_height + abs_margin_h + abs_hspace * (self.nrows - 1)
        self.horizontal_margins = abs_horizontal_margins / self.width
        self.vertical_margins = abs_vertical_margins / self.height

    def reset(self):
        """Resets the internal fig and gs to None"""
        self._fig = None
        self._gs = None

    def fig(self):
        """Get the underlying figure object"""
        if self._fig is None:
            self._fig = plt.figure(figsize=(self.width, self.height))
        return self._fig

    def gs(self):
        """Get the underlying GridSpec object"""
        if self._gs is None:
            self._gs = GridSpec(
                figure=self._fig,
                nrows=self.nrows,
                ncols=self.ncols,
                left=self.horizontal_margins[0],
                bottom=self.vertical_margins[0],
                right=1.0 - self.horizontal_margins[1],
                top=1.0 - self.vertical_margins[1],
                hspace=self.hspace,
                wspace=self.wspace,
                width_ratios=self.width_ratios,
                height_ratios=self.height_ratios,
            )
        return self._gs

    def annotate(self, ax, text, pos=[0, 0.98], fontsize=8, **kwargs):
        """Annotate an ax with some text. Wrapper around ax.text.

        Args:
            ax (plt.ax): the ax
            text (str): the text
            pos (list, optional): position. Defaults to [0, 0.98].
            fontsize (int, optional): fontsize. Defaults to 8.
        """
        ax.text(
            *pos,
            text,
            fontsize=fontsize,
            horizontalalignment="left",
            verticalalignment="top",
            transform=ax.transAxes,
            **kwargs,
        )

    def add_box_around_image(self, ax, axes_image, **kwargs):
        """Adds a box patch around an axis.

        Args:
            ax (plt.ax): The ax
            axes_image (_type_): The image

        Returns:
            FancyBboxPatch: Returns the box patch
        """
        extent = axes_image.get_extent()
        left, right, bottom, top = extent
        width = right - left
        height = top - bottom
        fancy = FancyBboxPatch((left, bottom), width, height, **kwargs)
        ax.add_patch(fancy)
        return fancy

    @staticmethod
    def open_image(path):
        return np.array(Image.open(path))

    @staticmethod
    def replace_background_color(image, replacement_color, background_color=None):
        """Replaced the backgroudn color of an image, specified as a numpy array.

        Args:
            image (np.Array): The image array
            replacement_color (the color): The color wich replaces the background color.
            background_color (the background color, optional): The backgroudn color. If None it is inferred from the corners. Defaults to None.

        Returns:
            np.Array: the new image array
        """

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

        N_CHANNELS_BG = len(replacement_color)

        # If the background color has more channels than the picture, we need to introduce the alpha channel
        if N_CHANNELS_BG > N_CHANNELS and replacement_color[-1] != 1.0:
            image_copy = np.ones(shape=(image_shape[0], image_shape[1], 4))
            image_copy[:, :, :3] = image
            image = image_copy
            background_color = [
                *background_color,
                1.0,
            ]  # Extend the background color with the alpha channel
        elif N_CHANNELS_BG < N_CHANNELS:
            replacement_color = [*replacement_color, 1.0]

        indices = np.argwhere(np.all(image[:, :] == background_color, axis=2))
        image[indices[:, 0], indices[:, 1], :] = replacement_color
        return image

    @staticmethod
    def crop_to_content(image, background_color=None, replace_background_color=None):
        """Crops an image array to its content."""
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

    @staticmethod
    def crop(image, width, height=None):
        """Crops an image array to a give width and height (towards the center)"""
        if height is None:
            height = image.shape[0]
        o = [int(image.shape[0] / 2), int(image.shape[1] / 2)]
        lower_height = o[0] - int(height / 2)
        lower_width = o[1] - int(width / 2)
        upper_width = lower_width + width
        upper_height = lower_height + height
        return image[lower_height:upper_height, lower_width:upper_width, :]

    def create_inset_axis(
        self,
        containing_ax,
        rel_width=0.5,
        rel_height=0.5,
        margin_x=0.0,
        margin_y=0.0,
        x_align="left",
        y_align="bottom",
    ):
        """Creates an axis for an inset"""
        pos = containing_ax.get_position(self._fig)

        w = pos.x1 - pos.x0
        h = pos.y1 - pos.y0

        def helper(old_var0, old_var1, align, wh, rel_width_height, margin_xy):
            if align == "left" or align == "bottom":
                new_var0 = old_var0 + margin_xy * rel_width_height * wh
                new_var1 = new_var0 + rel_width_height * wh
            elif align == "right" or align == "top":
                new_var1 = old_var1 - margin_xy * rel_width_height * wh
                new_var0 = new_var1 - rel_width_height * wh
            elif align == "center":
                new_var0 = old_var0 + 0.5 * (1.0 - rel_width_height) * wh
                new_var1 = old_var1 - 0.5 * (1.0 - rel_width_height) * wh
            else:
                raise Exception(f"unknown align: {align}")
            return new_var0, new_var1

        pos.x0, pos.x1 = helper(pos.x0, pos.x1, x_align, w, rel_width, margin_x)
        pos.y0, pos.y1 = helper(pos.y0, pos.y1, y_align, h, rel_height, margin_y)

        return self._fig.add_axes(pos)

    @staticmethod
    def image_to_ax(ax, image):
        import os

        if isinstance(image, str):
            if os.path.exists(image):
                image = Paper_Plot.open_image(image)
            else:
                raise Exception(f"`{image}` does not exist")

        ax.tick_params(
            axis="both", which="both", bottom=0, left=0, labelbottom=0, labelleft=0
        )
        ax.set_facecolor([0, 0, 0, 0])
        for k, s in ax.spines.items():
            s.set_visible(False)

        return ax.imshow(image)

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
