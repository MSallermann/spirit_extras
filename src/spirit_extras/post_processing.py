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


def hopfion_normal(spin_system, background_direction = [0,0,1], background_angle_diff = np.pi/4):
    """ Try to compute the center and normal of a Hopfion"""
    bg = np.array(background_direction)

    spins     = spin_system.spins
    positions = spin_system.positions

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

def compute_pre_image(positions, spins, pre_image_spin, angle_tolerance=0.1, n_neighbours=20):

    ## BEGIN
    def find_best_neighbour(neighbour_indices, distances, positions, last_segment):
        """iterate over the neighbours and find the best one"""
        segments = np.zeros(shape = (len(neighbour_indices),3))

        i_best         = 0
        idx_neigh_best = 0
        angle_min      = 999

        for i,idx_neigh in enumerate(neighbour_indices):
            # Compute the test segment
            test_segment = positions[ idx_neigh ] - positions[ pre_image[-1] ]
            test_segment /= np.linalg.norm( test_segment )
            segments[i] = test_segment

            # Find the segment that gives the smoothest path (smallest angle to the previous segment)
            angle = np.arccos( np.dot( last_segment, test_segment ) )
            if np.isnan(angle):
                angle = 0

            if angle < angle_min:
                angle_min = angle
                idx_neigh_best = idx_neigh
                i_best = i

        return i_best, idx_neigh_best, segments

    def exclude_indices(neighbour_indices, distances, segments, segment_best, distance_best):
        """Finds the neighbour_indices which should be excluded from subsequen pre-image searches"""
        in_path_direction = [ np.dot(segment_best, s) > 0 for s in segments ]
        less_distant      = distances <= distance_best
        excluded_indices  = neighbour_indices[ np.logical_and(in_path_direction, less_distant) ]
        return excluded_indices
    ## END

    from scipy.spatial import KDTree

    pre_image_spin = np.array( pre_image_spin ) / np.linalg.norm( pre_image_spin )

    angles = np.arccos( np.dot(spins, pre_image_spin) )

    idx_converter = np.array(range( len(spins) ) )[ angles < angle_tolerance ]
    positions     = positions[ angles < angle_tolerance ]
    spins         = spins[ angles < angle_tolerance ]

    if len(spins) < n_neighbours:
        raise Exception("Not enough spins match pre-image spin")

    angles = np.arccos( np.dot(spins, pre_image_spin) )

    tree = KDTree(positions)

    # First two points of the preimage
    angles    = np.arccos( np.dot(spins, pre_image_spin) ) # angles between spins and pre-image-spin
    idx_first = np.argmin(angles)

    # First spin is the one that best matches the pre-image-spin
    distances, neighbour_indices = tree.query( positions[idx_first], k=n_neighbours )
    neighbour_indices            = neighbour_indices[1:] # remove the point itself (distance zero from the distances and the neighbour indices)
    distances                    = distances[1:]
    angles_first_neigbours       = angles[ neighbour_indices ]

    # Second spin is the one that, among the neighbours of the first spin, best matches the pre-image-spin
    idx_second = neighbour_indices[ np.argmin( angles_first_neigbours ) ]

    pre_image    = [ idx_first, idx_second ]
    idx_excluded = [ idx_second ] # indices that are excluded from the pre-image search

    last_segment = positions[ pre_image[-1] ] - positions[ pre_image[-2] ]
    distance_best = np.linalg.norm( last_segment )
    last_segment /= distance_best # direction of the last segment of the pre-image

    segments = np.zeros(shape = (len(neighbour_indices), 3))
    for i,idx_neigh in enumerate(neighbour_indices):
        segments[i] = positions[ idx_neigh ] - positions[ pre_image[0] ] # segment from idx_first to neighbour_index
        segments[i] /= np.linalg.norm( segments[i] )

    # Update exclude indices for first segment
    idx_excluded.extend( exclude_indices( neighbour_indices, distances, segments, last_segment, distance_best ) )

    RUN = True
    while RUN:
        last_segment = positions[ pre_image[-1] ] - positions[ pre_image[-2] ]
        last_segment /= np.linalg.norm( last_segment ) # direction of the last segment of the pre-image

        distances, neighbour_indices = tree.query( positions[pre_image[-1]], k=n_neighbours )
        neighbour_indices            = neighbour_indices[1:] # remove the point itself (distance zero from the distances and the neighbour indices)
        distances                    = distances[1:]

        # Remove the excluded indices
        excluded          = np.isin(neighbour_indices, idx_excluded)
        neighbour_indices = neighbour_indices[~excluded]
        distances         = distances[~excluded]

        if len(neighbour_indices) == 0:
            RUN = False
            continue

        if pre_image[0] in neighbour_indices and np.dot(last_segment, positions[pre_image[0]] - positions[pre_image[-1]]) > 0:

            pre_image.append( pre_image[0] )
            RUN = False
            continue

        i_best, idx_neigh_best, segments = find_best_neighbour(neighbour_indices, distances, positions, last_segment)
        pre_image.append(idx_neigh_best)

        idx_excluded.extend( exclude_indices( neighbour_indices, distances, segments, segments[i_best], distances[i_best]  ) )

    return idx_converter[ pre_image ], positions[ pre_image ]