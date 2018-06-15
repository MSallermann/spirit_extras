import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
import numpy as np

from spirit import geometry, system

def plot_basis_cell(p_state, ax=None):
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


def plot_spins(p_state, ax = None, **kwargs):
    n_cell_atoms  = geometry.get_n_cell_atoms(p_state)
    n_cells       = geometry.get_n_cells(p_state)
    positions     = geometry.get_positions(p_state)
    spins         = system.get_spin_directions(p_state)

    x = positions[:,0]
    y = positions[:,1]
    u = spins[:,0]
    v = spins[:,1]
    C = spins[:,2]

    ax.quiver(x, y, u, v, C, pivot='mid',  angles='xy', scale_units='xy', scale=0.6)
    ax.scatter(x,y, marker = ".", color = "black", s = 1)