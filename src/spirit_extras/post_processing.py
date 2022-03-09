import numpy as np
from scipy.optimize import minimize

def get_linecut(p_state, x0, x1):
    pass

def get_sublattice_magnetisation(p_state, ibasis):
    pass

def interpolate_spins(p_state):
    pass

def skyrmion_radius(p_state):
    pass

def skyrmion_circularity(p_state):
    pass

def hopfion_normal(p_state, background_direction = [0,0,1], background_angle_diff = np.pi/4):
    from spirit import system, geometry
    bg = np.array(background_direction)

    spins     = system.get_spin_directions(p_state)
    positions = np.array(geometry.get_positions(p_state))

    background_mask  = [ np.arccos(np.dot( bg, s )) > background_angle_diff for s in spins ]
    masked_positions = positions[background_mask]
    center           = np.mean( masked_positions, axis = 0)

    radial_directions = masked_positions - center

    def cost_function(angles):
        phi    = angles[0]
        theta  = angles[1]
        normal = np.array( [np.cos(phi) * np.sin(theta), np.sin(phi) * np.sin(theta), np.cos(theta)] )
        result = np.sum( [ (np.dot(normal, r))**2  for r in radial_directions ] )
        return result

    res = minimize(cost_function, x0=[0.5, 0.5], tol=1e-12)

    angles_opt = res.x
    phi        = angles_opt[0]
    theta      = angles_opt[1]
    normal_opt = np.array( [np.cos(phi) * np.sin(theta), np.sin(phi) * np.sin(theta), np.cos(theta)] )

    return center, normal_opt