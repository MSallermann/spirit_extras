import matplotlib.pyplot as plt
from   matplotlib.patches import Polygon
from   matplotlib.collections import PatchCollection
from   matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
import matplotlib as mpl
import numpy as np
import os

class Paper_Plot:
    # Settings
    cm = 1/2.54

    # Annotations
    offset_u = 1.5*np.array([0,10])
    offset_r = 1.5*np.array([10,0])
    offset_l = -1.5*offset_r
    offset_d = -1.5*offset_u
    offset_ur = (offset_r + offset_u) / np.sqrt(2)
    offset_ul = (offset_l + offset_u) / np.sqrt(2)
    offset_dr = (offset_r + offset_d) / np.sqrt(2)
    offset_dl = (offset_l + offset_d) / np.sqrt(2)

    offset_dict = {
        "u" : offset_u,
        "r" : offset_r,
        "l" : offset_l,
        "d" : offset_d,
        "ur" : offset_ur,
        "ul" : offset_ul,
        "dr" : offset_dr,
        "dl" : offset_dl
    }

    def __init__(self, width) -> None:
        mpl.rcParams["font.size"]        = 8 #'dejavusans' (default),
        mpl.rcParams["font.family"]      = "serif" #'dejavusans' (default),
        mpl.rcParams["mathtext.fontset"] = "dejavuserif" #'dejavusans' (default),
        plt.rc('xtick', labelsize=8)
        plt.rc('ytick', labelsize=8)
        plt.rc('axes',  labelsize=8)

        self.annotate_offset_scale = 1

        self.annotate_letter = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        # self.annotate_letter = "ABCDEFGHIJKLMNOPQRSTUVWXYZ".lower()
        # self.annotate_letter = ["I", "II", "III", "IV", "V", "VI", "VII", "VIII", "IX", "X"]
        # self.annotate_letter = ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10"]

        self.width  = width
        self.height = width

        self.ncols              = 1
        self.nrows              = 1
        self.horizontal_margins = [0.1, 0.1]
        self.vertical_margins   = [0.1, 0.1]
        self.wspace             = 0
        self.hspace             = 0

        self.width_ratios  = None
        self.height_ratios = None

        self._fig = None
        self._gs  = None

        # self.annotate_increment = 0
        self.annotation_dict = {}

    def height_from_aspect_ratio(self, aspect_ratio):
        rel_margin_w = sum(self.horizontal_margins)
        rel_space_w  = self.wspace * (1-rel_margin_w)/self.ncols

        rel_margin_h = sum(self.vertical_margins)
        rel_space_h  = self.hspace * (1-rel_margin_h)/self.nrows

        self.height = self.width/aspect_ratio * (1 - rel_margin_w - rel_space_w) / (1 - rel_margin_h - rel_space_h)

    def fig(self):
        self._fig = plt.figure(figsize = (self.width, self.height))
        return self._fig

    def gs(self):
        self._gs = GridSpec(figure=self._fig, nrows=self.nrows, ncols=self.ncols, left=self.horizontal_margins[0], bottom=self.vertical_margins[0], right=1-self.horizontal_margins[1], top=1-self.vertical_margins[1], hspace=self.hspace, wspace=self.wspace, width_ratios=self.width_ratios, height_ratios=self.height_ratios) 
        return self._gs

    def annotate(self, ax, text, pos = [0,0.98], fontsize=8):
        ax.text(*pos, text, fontsize=fontsize, horizontalalignment='left', verticalalignment='top', transform=ax.transAxes)

    def crop(self, image, width, height=None):
        if height is None:
            height = image.shape[1]

        o             = [int(image.shape[0]/2), int(image.shape[1]/2)]
        lower_height  = o[0] - int(height/2)
        lower_width   = o[1] - int(width/2)
        upper_width = lower_width + width
        upper_height = lower_height + height
        return image[lower_height:upper_height, lower_width:upper_width, :]

    def image_to_ax(self, ax, image):
        import os
        if os.path.exists(image):
            image = plt.imread(image)
        ax.imshow(image)
        ax.tick_params(axis='both', which='both', bottom=0, left=0, labelbottom=0, labelleft=0)
        ax.set_facecolor([0,0,0,0])
        for k,s in ax.spines.items():
            s.set_visible(False)

    def spine_axis(self, subplotspec, color="black", which=["left", "right", "top", "bottom"]):
        a = self._fig.add_axes( subplotspec.get_position(self._fig), zorder=99)
        a.set_facecolor([0,0,0,0])
        a.tick_params(axis='both', which='both', bottom=0, left=0, labelbottom=0, labelleft=0)
        for k in which:
            s = a.spines[k]
            s.set_visible(True)
            s.set_color(color)
        return a

    def row(self, row_idx, sl=slice(None, None, None), gs=None):
        if gs is None:
            gs = self._gs

        col_indices = range(gs.ncols)[sl]
        return [ self._fig.add_subplot( gs[row_idx, col_idx] ) for col_idx in col_indices ]

    def col(self, col_idx, sl=slice(None, None, None), gs=None):
        if gs is None:
            gs = self._gs

        row_indices = list(range(gs.nrows)[sl])
        return [ self._fig.add_subplot(gs[row_idx, col_idx] ) for row_idx in row_indices ]

    def xy_text_auto(self, ax, xy, deriv, scale=15):
        trans_deriv   =  ax.transData.transform( [[0,0],[1,deriv]] ) # transform from data coordinates to display coordinates
        display_deriv = (trans_deriv[1,1] - trans_deriv[0,1]) / (trans_deriv[1,0] - trans_deriv[0,0])

        # display_deriv = trans_deriv[1]/trans_deriv[0] # transform to display derivative
        direction = np.array([display_deriv, -1])
        direction = direction/np.linalg.norm(direction)

        # Where do we hit the upper ylim in display coords
        lim_lower = ax.transData.transform( [ax.get_xlim()[0], ax.get_ylim()[0]])
        lim_upper = ax.transData.transform( [ax.get_xlim()[1], ax.get_ylim()[1]])
        xy = ax.transData.transform(xy)

        intersect_upper = (lim_upper - xy) / direction
        dist_upper      = np.min( np.abs( intersect_upper ) )
        sign_upper      = np.sign(intersect_upper[0])

        intersect_lower = (lim_lower - xy) / direction
        dist_lower      = np.min( np.abs( intersect_lower ) )
        sign_lower      = np.sign(intersect_lower[0])

        if dist_upper > dist_lower:
            return scale * sign_upper * direction
        else:
            return scale * sign_lower * direction

    def annotate_graph(self, ax, xy, xy_text, text=None, key="key1", offset_scale=1):

        if not key is None:
            if not key in self.annotation_dict:
                self.annotation_dict[key] = {"annotate_increment" : 0, "annotation_list" : []}
        elif text is None:
            raise Exception("Need to specify text if key is None")

        arrowprops = dict(arrowstyle="-")

        if text is None:
            text = self.annotate_letter[self.annotation_dict[key]["annotate_increment"]]
            self.annotation_dict[key]["annotate_increment"]  += 1

        if type(xy_text) is str:
            xy_text = Paper_Plot.offset_dict[xy_text.lower()]

        if not key is None:
            self.annotation_dict[key]["annotation_list"].append( [xy, text] )

        ax.annotate(text, xy, xy_text * offset_scale, arrowprops = arrowprops, verticalalignment="center", horizontalalignment="center", textcoords="offset points")

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
    n_cell_atoms  = geometry.get_n_cell_atoms(p_state)
    n_cells       = geometry.get_n_cells(p_state)
    positions     = geometry.get_positions(p_state)

    def idx(i,a,b,c):
        return i + n_cell_atoms * (a + n_cells[0] * (b + n_cells[1] * c))

    # Draw the outlines of the basis cell
    lattice_points = np.array([ positions[idx(0,0,0,0)], positions[idx(0,1,0,0)], positions[idx(0,1,1,0)], positions[idx(0,0,1,0)] ])

    patch=[Polygon(lattice_points[:,:2])]
    pc = PatchCollection(patch, linestyle="--", facecolor = [0,0,0,0], edgecolor = [0,0,0,1])
    ax.add_collection(pc)
    ax.scatter(positions[:n_cell_atoms,0], positions[:n_cell_atoms,1], color="red", s=100)
    ax.set_xlabel(r"$x\;\;[\AA]$")
    ax.set_ylabel(r"$y\;\;[\AA]$")

def set_kwarg_if_not_there(kwarg_dict, key, value):
    if not key in kwarg_dict:
        kwarg_dict[key] = value

def plot_spins_2d(spin_system, ax, col_xy = [0,1], x=None, y=None, u=None, v=None, c=None, **kwargs):

    set_kwarg_if_not_there(kwargs, "lw", 0.5)
    set_kwarg_if_not_there(kwargs, "units", "xy")
    set_kwarg_if_not_there(kwargs, "edgecolor", "black")
    set_kwarg_if_not_there(kwargs, "scale", 2)
    set_kwarg_if_not_there(kwargs, "width", 0.13)
    set_kwarg_if_not_there(kwargs, "pivot", "mid")
    set_kwarg_if_not_there(kwargs, "cmap", "seismic")
    set_kwarg_if_not_there(kwargs, "alpha", 1)
    set_kwarg_if_not_there(kwargs, "clim", (-1,1))

    if (not (x and y)):
        x = spin_system.positions[:,col_xy[0]]
        y = spin_system.positions[:,col_xy[1]]

    if (not (u and v and c)):
        u = spin_system.spins[:,col_xy[0]]
        v = spin_system.spins[:,col_xy[1]]
        color_column = [ i for i in [0,1,2] if i not in col_xy ]
        c = spin_system.spins[:,color_column]

    ax.set_aspect("equal")

    cb = ax.quiver(x,y,u,v,c, **kwargs)
    ax.scatter(x,y, marker = ".", color = "black", s = 1)
    ax.set_xlabel(r"$x~[\AA]$")
    ax.set_ylabel(r"$y~[\AA]$")
    colorbar(cb, label=r"$\mathbf{m}_z$")

def plot_energy_path(energy_path, ax, normalize_reaction_coordinate = False, kwargs_discrete={}, kwargs_interpolated={}, plot_interpolated=True):

    set_kwarg_if_not_there(kwargs_discrete, "markeredgecolor", "black")
    set_kwarg_if_not_there(kwargs_discrete, "marker", "o")
    set_kwarg_if_not_there(kwargs_discrete, "ls", "None")
    set_kwarg_if_not_there(kwargs_discrete, "color", "C0")

    set_kwarg_if_not_there(kwargs_interpolated, "lw", 1.75)
    set_kwarg_if_not_there(kwargs_interpolated, "marker", "None")
    set_kwarg_if_not_there(kwargs_interpolated, "ls", "-")
    set_kwarg_if_not_there(kwargs_interpolated, "color", kwargs_discrete["color"])

    E0 = energy_path.total_energy[-1]

    if(len(energy_path.interpolated_reaction_coordinate) > 0 and plot_interpolated):
        Rx_int = np.array(energy_path.interpolated_reaction_coordinate)
        if normalize_reaction_coordinate:
            Rx_int = Rx_int / Rx_int[-1]
        ax.plot(Rx_int, np.asarray(energy_path.interpolated_total_energy) - E0, **kwargs_interpolated )

    Rx = np.array(energy_path.reaction_coordinate)
    if normalize_reaction_coordinate:
        Rx = Rx / Rx[-1]
    ax.plot(Rx, np.asarray(energy_path.total_energy) - E0, **kwargs_discrete)

    ax.set_xlabel( "reaction coordinate [arb. units]" )
    ax.set_ylabel( r"$\Delta E$ [meV]" )

def get_rgba_colors(spins, opacity=1.0, cardinal_a=np.array([1,0,0]), cardinal_b=np.array([0,1,0]), cardinal_c=np.array([0,0,1])):
    """Returns a colormap in the matplotlib format (an Nx4 array)"""
    import math

    # Annoying OpenGl functions

    def atan2(y, x):
        if x==0.0:
            return np.sign(y) * np.pi / 2.0
        else:
            return np.arctan2(y, x)

    def fract(x):
        return x - math.floor(x)

    def mix(x,y,a):
        return x * (1.0-a) + y * a

    def clamp(x, minVal, maxVal):
        return min(max(x, minVal), maxVal)

    def hsv2rgb(c):
        K = [1.0, 2.0 / 3.0, 1.0 / 3.0, 3.0]

        px = abs( fract(c[0] + K[0]) * 6.0 - K[3] )
        py = abs( fract(c[0] + K[1]) * 6.0 - K[3] )
        pz = abs( fract(c[0] + K[2]) * 6.0 - K[3] )
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
        hue          = atan2( projection_x, projection_y ) / (2*np.pi)

        saturation   = projection_z

        if saturation > 0.0:
            saturation = 1.0 - saturation
            value = 1.0
        else:
            value = 1.0 + saturation
            saturation = 1.0

        rgba.append( [*hsv2rgb( [hue, saturation, value] ), opacity] )

    return rgba