import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
import numpy as np

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
    import colorsys

    rgba = []

    for spin in spins:
        phi_offset = 0
        projection_x = cardinal_a.dot(spin)
        projection_y = cardinal_b.dot(spin)
        projection_z = cardinal_c.dot(spin)
        hue          = (np.arctan2( projection_x, projection_y ) + np.pi) / (2*np.pi)
        saturation   = (cardinal_c.dot(spin) + 1.0) / 2.0
        value        = np.sqrt( 1 - saturation**2)
        rgba.append( [*colorsys.hsv_to_rgb(hue, saturation, value), opacity] )

    return rgba