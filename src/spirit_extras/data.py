import numpy as np

class Spin_System:

    def __init__(self):
        self.positions       = []
        self.spins           = []
        self.n_cells         = [0,0,0]
        self.n_cell_atoms    = 0
        self.basis           = [[0,0,0]]
        self.bravais_vectors = [[0,0,0], [0,0,0], [0,0,0]]
        self.unordered       = False

    def __getitem__(self, key):
        sliced_spin_system           = Spin_System()
        sliced_spin_system.positions = self.positions[key]
        sliced_spin_system.spins     = self.spins[key]
        sliced_spin_system.unordered = True # For a general slice we do not necessarily retain the lattice structure
        return sliced_spin_system

    def copy(self):
        """Make a shallow copy of spin_system"""
        copy = Spin_System()
        copy.positions       = self.positions # This will create a copy of the 'metadata' like shape, but not the underlying field
        copy.spins           = self.spins # This will create a copy of the 'metadata' like shape, but not the underlying field
        copy.n_cell_atoms    = self.n_cell_atoms
        copy.n_cells         = np.array(self.n_cells)
        copy.basis           = np.array(self.basis)
        copy.bravais_vectors = np.array(self.bravais_vectors)
        copy.unordered       = self.unordered
        return copy

    def deepcopy(self):
        """Make a deep copy of spin system"""
        copy = Spin_System()
        copy.positions       = np.array(self.positions)
        copy.spins           = np.array(self.spins)
        copy.n_cell_atoms    = self.n_cell_atoms
        copy.n_cells         = np.array(self.n_cells)
        copy.basis           = np.array(self.basis)
        copy.bravais_vectors = np.array(self.bravais_vectors)
        copy.unordered       = self.unordered
        return copy

    def is_flat(self):
        """Return true if spin_system is flat"""
        return len(self.positions.shape) == 2

    def flatten(self):
        """Flatten the spin system"""
        self.positions = np.reshape(self.positions, (self.nos(),3), order="F")
        self.spins     = np.reshape(self.spins, (self.nos(),3), order="F")

    def flattened(self):
        """Return a flattend view into the spin system"""
        if self.is_flat():
            return self
        else:
            temp = self.copy()
            temp.positions = np.reshape(self.positions, (self.nos(),3), order="F")
            temp.spins     = np.reshape(self.spins, (self.nos(),3), order="F")
            return temp

    def shape(self):
        """Shape the spin system"""
        if self.unordered:
            raise Exception("Cannot shape an unordered system")

        self.positions = np.reshape(self.positions, (self.n_cell_atoms, self.n_cells[0], self.n_cells[1], self.n_cells[2], 3), order="F")
        self.spins     = np.reshape(self.spins, (self.n_cell_atoms, self.n_cells[0], self.n_cells[1], self.n_cells[2], 3), order="F")

    def shaped(self):
        """Return a shaped view into the spin system"""
        if self.unordered:
            raise Exception("Cannot shape an unordered system")

        if not self.is_flat():
            return self
        else:
            temp = self.copy()
            temp.positions = np.reshape(self.positions, (self.n_cell_atoms, self.n_cells[0], self.n_cells[1], self.n_cells[2], 3), order="F")
            temp.spins     = np.reshape(self.spins, (self.n_cell_atoms, self.n_cells[0], self.n_cells[1], self.n_cells[2], 3), order="F")
            return temp

    def nos(self):
        if self.unordered:
            return len(self.spins)
        else:
            return self.n_cells[0] * self.n_cells[1] * self.n_cells[2] * self.n_cell_atoms

    def center(self):
        return np.mean(self.flattened().positions, axis=0)

    def a_slice(self, val):
        result                 = self.shaped()[:,val,:,:,:]
        result.n_cells         = [1, self.n_cells[1], self.n_cells[2]]
        result.bravais_vectors = self.bravais_vectors
        result.n_cell_atoms    = self.n_cell_atoms
        result.unordered       = False
        return result.flattened()

    def b_slice(self, val):
        result                 = self.shaped()[:,:,val,:,:]
        result.n_cells         = [self.n_cells[0], 1, self.n_cells[2]]
        result.bravais_vectors = self.bravais_vectors
        result.n_cell_atoms    = self.n_cell_atoms
        result.unordered       = False
        return result.flattened()

    def c_slice(self, val):
        result                 = self.shaped()[:,:,:,val,:]
        result.n_cells         = [self.n_cells[0], self.n_cells[1], 1]
        result.bravais_vectors = self.bravais_vectors
        result.n_cell_atoms    = self.n_cell_atoms
        result.unordered       = False
        return result.flattened()

def spin_system_from_p_state(p_state, idx_image=-1, copy=False):
    from spirit import geometry, system
    spin_system = Spin_System()
    spin_system.positions     = geometry.get_positions(p_state, idx_image = idx_image)
    spin_system.spins         = system.get_spin_directions(p_state, idx_image = idx_image)
    spin_system.n_cell_atoms  = geometry.get_n_cell_atoms(p_state)
    spin_system.n_cells       = geometry.get_n_cells(p_state)

    if copy:
        spin_system.positions    = np.array(spin_system.positions   )
        spin_system.spins        = np.array(spin_system.spins       )
        spin_system.n_cell_atoms = np.array(spin_system.n_cell_atoms)
        spin_system.n_cells      = np.array(spin_system.n_cells     )

    return spin_system

class energy_path:

    def __init__(self):
        self.reaction_coordinate  = []
        self.total_energy         = []
        self.energy_contributions = {}
        self.interpolated_reaction_coordinate = []
        self.interpolated_total_energy = []
        self.interpolated_energy_contributions = {}

    def idx_sp(self):
        return np.argmax(self.total_energy)

    def noi(self):
        return len(self.reaction_coordinate)

    def split(self, idx_0, idx_1):
        #TODO: interpolated quantities
        split_path = energy_path()
        split_path.reaction_coordinate = np.array(self.reaction_coordinate[idx_0:idx_1])
        split_path.total_energy        = np.array(self.total_energy[idx_0:idx_1])
        return split_path

    def slope(self):
        result = np.gradient(self.total_energy, self.reaction_coordinate)
        return result

    def interpolated_slope(self):
        result = np.gradient(self.interpolated_total_energy, self.interpolated_reaction_coordinate)
        return result

    def curvature(self):
        result = np.gradient(self.slope(), self.reaction_coordinate)
        return result

    def interpolated_curvature(self):
        result = np.gradient(self.interpolated_slope(), self.interpolated_reaction_coordinate)
        return result

    def n_interpolated(self):
        return int( ( len(self.interpolated_reaction_coordinate) - self.noi() ) / (self.noi() - 1) )

    def barrier(self):
        return np.max(self.total_energy) - self.total_energy[0]

def energy_path_from_p_state(p_state):
    #TODO: contributions
    from spirit import chain
    result = energy_path()
    result.reaction_coordinate              = chain.get_reaction_coordinate(p_state)
    result.total_energy                     = chain.get_energy(p_state)
    result.interpolated_reaction_coordinate = chain.get_reaction_coordinate_interpolated(p_state)
    result.interpolated_total_energy        = chain.get_energy_interpolated(p_state)

    if(result.interpolated_reaction_coordinate[-1] == 0): # Quick check if the interpolated quantities have been computed
        result.interpolated_reaction_coordinate = []
        result.interpolated_total_energy = []

    return result