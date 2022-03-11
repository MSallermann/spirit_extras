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

def skyrmion_circularity( positions, spins, center, z_value = 0, background_direction=[0,0,1], max_radius = 1, cutoff_ring = 0.33, N_Segments=360, N_rho=100 ):
    """ Try to compute the circularity of a Skyrmion"""

    from spirit_extras import data
    from scipy.optimize import curve_fit, root
    from scipy.interpolate import griddata, interp1d

    # Cut out the positions that we consider the circumference of the skyrmion to lie within
    ring_mask      = np.logical_and( np.abs( spins[:,2] - z_value ) <= cutoff_ring, np.linalg.norm( positions - center, axis=1) <= max_radius )
    positions_ring = positions[ ring_mask ]
    spins_ring     = spins[ ring_mask ]

    if len(positions_ring) <= 2:
        print("Too few positions in ring_mask. Try to increase cutoff_ring")

    # List of radii for each angular segment, initialized with the average_radius
    rho_list = np.ones(N_Segments) * max_radius/2.0

    # List of the points lieing onf the circumference, to be filled later
    circ_grid = np.zeros((N_Segments,2))

    circumference  = 0
    area           = 0

    for i in range(N_Segments):
        phi = 2*np.pi/N_Segments * i

        # A series of points extending along the radial direction for each segment
        rho_temp   = np.linspace(0, max_radius, N_rho)
        rho_2dgrid = np.array( [ np.array( [np.cos(phi), np.sin(phi)] ) * rho + center[:2] for rho in rho_temp ] )

        # Interpolate the spins z component onto the circular grid
        z_2dgrid = griddata( (positions_ring[:,0], positions_ring[:,1]), spins_ring[:,2], (rho_2dgrid[:,0], rho_2dgrid[:,1]), method="linear" )

        rho_max = np.max(rho_temp[~np.isnan(z_2dgrid)])
        rho_min = np.min(rho_temp[~np.isnan(z_2dgrid)])

        rho_temp = rho_temp[~np.isnan(z_2dgrid)]
        z_2dgrid = z_2dgrid[~np.isnan(z_2dgrid)]

        # Interpolate the z component of the spins onto the radial direction
        z_func   = interp1d(rho_temp, z_2dgrid, kind = "linear")

        # Find where the z component crosses our target value
        def obj_fun(rho):
            if rho<rho_min:
                return -1.0
            if rho>rho_max:
                return 1.0
            return z_func(rho) - z_value

        root_res          = root( obj_fun, x0 = rho_list[i-1] )

        rho = root_res.x[0]
        rho_list[i]  = rho
        circ_grid[i] = center[:2] + np.array([np.cos(phi) * rho, np.sin(phi)*rho])

        if i > 0:
            circumference += np.linalg.norm( circ_grid[i]- circ_grid[i-1] ) # Increment the circumference
            area          += np.linalg.norm( np.cross(np.array([np.cos(phi) * rho, np.sin(phi)*rho]), ( circ_grid[i] - circ_grid[i-1]))) / 2 # Increment the area

    # Compute the circularity
    circularity    = 4 * np.pi *area/(circumference**2)
    average_radius = np.mean(rho_list)

    result = {
        "area" : area,
        "circumference" : circumference,
        "circularity" : circularity,
        "average_radius" : average_radius,
        "rho_list" : rho_list,
        "circ_grid" : circ_grid
    }

    return result


def hopfion_normal(p_state, background_direction = [0,0,1], background_angle_diff = np.pi/4):
    """ Try to compute the center and normal of a Hopfion"""
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