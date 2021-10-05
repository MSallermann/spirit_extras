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

    set_kwarg_if_not_there(kwargs, "lw", 1)
    set_kwarg_if_not_there(kwargs, "units", "xy")
    set_kwarg_if_not_there(kwargs, "edgecolor", "black")
    set_kwarg_if_not_there(kwargs, "scale", 2)
    set_kwarg_if_not_there(kwargs, "width", 0.13)
    set_kwarg_if_not_there(kwargs, "pivot", "mid")
    set_kwarg_if_not_there(kwargs, "cmap", "seismic")
    set_kwarg_if_not_there(kwargs, "alpha", 0.65)
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