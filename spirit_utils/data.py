import numpy as np

class Spin_System:
    positions       = []
    spins           = []
    n_cells         = [0,0,0]
    n_cell_atoms    = 0
    basis           = [[0,0,0]]
    bravais_vectors = [[0,0,0],[0,0,0],[0,0,0]]
    unordered = False

    def __getitem__(self, key):
        sliced_spin_system = Spin_System()
        sliced_spin_system.positions = self.positions[key]
        sliced_spin_system.spins     = self.spins[key]
        sliced_spin_system.unordered = True # For a general slice we do not necessarily retain the lattice structure
        return sliced_spin_system

    def copy(self):
        copy = Spin_System()
        copy.positions       = np.array(self.positions)
        copy.spins           = np.array(self.spins)
        copy.n_cells         = np.array(self.n_cells)
        copy.basis           = np.array(self.basis)
        copy.bravais_vectors = np.array(self.bravais_vectors)
        copy.unordered       = self.unordered
        return copy

    def is_flat(self):
        return len(self.positions.shape) == 2

    def flatten(self):
        self.positions = np.reshape(self.positions, (self.nos(),3), order="F")
        self.spins     = np.reshape(self.spins, (self.nos(),3), order="F")
        return self

    def shape(self):
        self.positions = np.reshape(self.positions, (self.n_cell_atoms, self.n_cells[0], self.n_cells[1], self.n_cells[2], 3), order="F")
        self.spins     = np.reshape(self.spins, (self.n_cell_atoms, self.n_cells[0], self.n_cells[1], self.n_cells[2], 3), order="F")
        return self

    def nos(self):
        return self.n_cells[0] * self.n_cells[1] * self.n_cells[2] * self.n_cell_atoms

    def center(self):
        was_flat = self.is_flat()
        if not was_flat:
            self.flatten()
        center =  np.mean(self.positions, axis=0)
        if not was_flat:
            self.shape()
        return center

    def a_slice(self, val):
        was_flat = self.is_flat()
        if was_flat:
            self.shape()

        result = self[:,val,:,:,:]
        result.n_cells = [1, self.n_cells[1], self.n_cells[2]]
        result.bravais_vectors = self.bravais_vectors
        result.n_cell_atoms = self.n_cell_atoms
        result.unordered = False

        if was_flat:
            self.flatten()

        return result.flatten()

    def b_slice(self, val):
        was_flat = self.is_flat()
        if was_flat:
            self.shape()

        result = self[:,:,val,:,:]
        result.n_cells = [self.n_cells[0], 1, self.n_cells[2]]
        result.bravais_vectors = self.bravais_vectors
        result.n_cell_atoms = self.n_cell_atoms
        result.unordered = False

        if was_flat:
            self.flatten()

        return result.flatten()

    def c_slice(self, val):
        was_flat = self.is_flat()
        if was_flat:
            self.shape()

        result = self[:,:,:,val,:]
        result.n_cells = [self.n_cells[0], self.n_cells[1], 1]
        result.bravais_vectors = self.bravais_vectors
        result.n_cell_atoms = self.n_cell_atoms
        result.unordered = False

        if was_flat:
            self.shape()

        return result.flatten()

def spin_system_from_p_state(p_state, copy=False):
    from spirit import geometry, system
    spin_system = Spin_System()
    spin_system.positions     = geometry.get_positions(p_state)
    spin_system.spins         = system.get_spin_directions(p_state)
    spin_system.n_cell_atoms  = geometry.get_n_cell_atoms(p_state)
    spin_system.n_cells       = geometry.get_n_cells(p_state)

    if copy:
        spin_system.positions    = np.array(spin_system.positions   )
        spin_system.spins        = np.array(spin_system.spins       )
        spin_system.n_cell_atoms = np.array(spin_system.n_cell_atoms)
        spin_system.n_cells      = np.array(spin_system.n_cells     )

    return spin_system

class energy_path:
    reaction_coordinate  = []
    total_energy         = []
    energy_contributions = {}
    interpolated_reaction_coordinate = []
    interpolated_total_energy = []
    interpolated_energy_contributions = {}

    def split(self, idx_0, idx_1):
        #TODO: interpolated quantities
        split_path = energy_path()
        split_path.reaction_coordinate = np.array(self.reaction_coordinate[idx_0:idx_1])
        split_path.total_energy        = np.array(self.total_energy[idx_0:idx_1])
        return split_path

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

    return result