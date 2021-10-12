import numpy as np

def relative_pos_to_cylindrical(relative_pos):
    rho = np.linalg.norm(relative_pos[:2])
    phi = np.arctan2(*relative_pos[:2])
    z = relative_pos[2]
    return rho, phi, z

def relative_pos_to_spherical(relative_pos):
    r = np.linalg.norm(relative_pos)
    phi = np.arctan2(*relative_pos[:2])
    theta = np.arccos(relative_pos[2])
    return r, phi, theta

def relative_pos_to_rectangular(relative_pos):
    x = relative_pos[0]
    y = relative_pos[1]
    z = relative_pos[2]
    return x,y,z

def _standard_filter(ib, a, b, c, rel_pos, *args):
    return True

def spherical_filter(radius):
    def _filter(ib, a, b, c, rel_pos):
        r, phi, theta = relative_pos_to_spherical(rel_pos)
        return r<radius
    return _filter

def cylindrical_filter(radius):
    def _filter(ib, a, b, c, rel_pos):
        rho, phi, z = relative_pos_to_cylindrical(rel_pos)
        return rho<radius
    return _filter


def cylindrical_configuration(spin_system, center, configuration_function, filter_function = _standard_filter, mode="replace"):
    spin_system.shape()

    for c in range(spin_system.n_cells[2]):
        for b in range(spin_system.n_cells[1]):
            for a in range(spin_system.n_cells[0]):
                for ib in range(spin_system.n_cell_atoms):

                    relative_pos = spin_system.positions[ib, a, b, c] - center

                    rho, phi, z = relative_pos_to_cylindrical(relative_pos)

                    spin = configuration_function(rho, phi, z)

                    if not filter_function(ib, a, b, c, relative_pos):
                        continue

                    if mode.lower() == "replace":
                        spin_system.spins[ib, a, b, c] = spin
                    elif mode.lower() == "add":
                        spin_system.spins[ib, a, b, c] += spin
                    spin_system.spins[ib, a, b, c] /= np.linalg.norm(spin_system.spins[ib, a, b, c])


def spherical_configuration(spin_system, center, configuration_function, filter_function = _standard_filter, mode="replace"):
    spin_system.shape()

    for c in range(spin_system.n_cells[2]):
        for b in range(spin_system.n_cells[1]):
            for a in range(spin_system.n_cells[0]):
                for ib in range(spin_system.n_cell_atoms):

                    relative_pos = spin_system.positions[ib, a, b, c] - center
                    r, phi, theta = relative_pos_to_spherical(relative_pos)
                    spin = configuration_function(r, phi, theta)

                    if not filter_function(ib, a, b, c, relative_pos):
                        continue

                    if mode.lower() == "replace":
                        spin_system.spins[ib, a, b, c] = spin
                    elif mode.lower() == "add":
                        spin_system.spins[ib, a, b, c] += spin
                    spin_system.spins[ib, a, b, c] /= np.linalg.norm(spin_system.spins[ib, a, b, c])


def rectangular_configuration(spin_system, center, configuration_function, filter_function = _standard_filter, mode="replace"):
    spin_system.shape()

    for c in range(spin_system.n_cells[2]):
        for b in range(spin_system.n_cells[1]):
            for a in range(spin_system.n_cells[0]):
                for ib in range(spin_system.n_cell_atoms):

                    relative_pos = spin_system.positions[ib, a, b, c] - center
                    x, y, z = relative_pos_to_rectangular(relative_pos)
                    spin = configuration_function(x, y, z)

                    if not filter_function(ib, a, b, c, relative_pos):
                        continue

                    if mode.lower() == "replace":
                        spin_system.spins[ib, a, b, c] = spin
                    elif mode.lower() == "add":
                        spin_system.spins[ib, a, b, c] += spin
                    spin_system.spins[ib, a, b, c] /= np.linalg.norm(spin_system.spins[ib, a, b, c])