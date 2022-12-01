import numpy as np


class Spin_System:
    def __init__(
        self,
        positions,
        spins,
        unordered=True,
        n_cells=None,
        basis=None,
        bravais_vectors=None,
    ):
        self.positions = np.asarray(positions)
        self.spins = np.asarray(spins)

        if self.positions.shape != self.spins.shape:
            raise Exception("Positions and spins have different shapes")

        # Check the shapes of spins and positions
        if not (
            (len(self.positions.shape) == 2 or len(self.positions.shape) == 5)
            and self.positions.shape[-1] == 3
        ):
            raise Exception(
                "Spins and positions must either have shape (nos,3) or (n_cell_atoms, n_cells[0], n_cells[1], n_cells[2], 3), but they have shape {}".format(
                    positions.shape
                )
            )

        self.unordered = unordered

        # If shaped positions and spins are given, we perform some consistency checks and try to infer n_cells, if not given
        if self.is_shaped():
            if self.unordered:
                raise Exception(
                    "Only arrays of shape (nos,3) can be used when unordered=True"
                )

            # Check if n_cells is consistent with the shape of the spin arrays
            if not n_cells is None:
                if self.positions.shape[1:-1] != (n_cells[0], n_cells[1], n_cells[2]):
                    raise Exception("Shape of positions/spins does not match n_cells")
            else:
                n_cells = np.array(self.positions.shape[1:-1])

            # Check if basis is consistent with the shape of the spin arrays
            # (Theoretically we could infer the basis as well)
            if not basis is None:
                if self.positions.shape[0] != len(basis):
                    raise Exception(
                        "Shape of positions/spins does not match length of basis"
                    )

        # If the system has order we need the lattice information
        if not self.unordered:
            for tmp in [n_cells, bravais_vectors]:
                if tmp is None:
                    raise Exception(
                        "For an ordered system 'n_cells' and 'bravais_vectors' has to be provided"
                    )

                self.n_cells = np.asarray(n_cells)
                if self.n_cells.shape != (3,):
                    raise Exception(
                        "n_cells has wrong shape. It should be (3), but is {}".format(
                            self.n_cells.shape
                        )
                    )

                if basis is None:
                    self.basis = np.array(
                        [[0, 0, 0]]
                    )  # default for single basis systems
                else:
                    self.basis = np.asarray(basis)

                if not self._check_shape(self.basis):
                    raise Exception("`basis` has wrong shape.")

                self.bravais_vectors = np.asarray(bravais_vectors)
                if not self._check_shape(self.basis):
                    raise Exception("`bravais_vectors` has wrong shape.")

    def require_order(func):
        from functools import wraps

        @wraps(func)
        def wrapper(self, *args, **kw):
            if self.unordered:
                raise Exception("This function requires on ordered system!")
            return func(self, *args, **kw)

        return wrapper

    def _check_shape(self, arr):
        if len(arr) == 0 or len(arr) == 1:
            return True
        if len(arr.shape) != 2 or arr.shape[-1] != 3:
            return False
        return True

    def __getitem__(self, key):
        sliced_spin_system = Spin_System(self.positions[key], self.spins[key])
        return sliced_spin_system

    @property
    @require_order
    def n_cell_atoms(self):
        return len(self.basis)

    @property
    def nos(self):
        if self.unordered:
            return len(self.spins)
        else:
            return (
                self.n_cells[0] * self.n_cells[1] * self.n_cells[2] * self.n_cell_atoms
            )

    @require_order
    def idx(self, ib, a, b, c):
        """Computes the linear idx from coords in the bravais lattice.

        Args:
            ib (int): idx of spin in basis cell
            a (int): first bravais translation index
            b (int): second bravais translation index
            c (int): third bravais translation index

        Returns:
            int: the linear index for the flattened spin/position arrays
        """

        return int(
            ib + self.n_cell_atoms * (a + self.n_cells[0] * (b + self.n_cells[1] * c))
        )

    @require_order
    def tupel(self, idx):
        """Computes the tupel of bravais indices from a linear idx.

        Args:
            idx (int): linear index

        Returns:
            list(int,int,int,int): list of bravais indices [idx_cell_atom, a, b, c]
        """
        idx_diff = idx

        maxVal = np.array([self.n_cell_atoms, *self.n_cells])
        tupel = [0, 0, 0, 0]

        div = np.prod(maxVal[:-1])

        for i in range(len(maxVal) - 1, 0, -1):
            tupel[i] = int(idx_diff / div)
            idx_diff -= tupel[i] * div
            div /= maxVal[i - 1]

        tupel[0] = int(idx_diff / div)

        return tupel

    def copy(self):
        """Make a shallow copy of spin_system"""

        # Need these field if not unordered
        if not self.unordered:
            _n_cells = np.array(self.n_cells)
            _basis = np.array(self.basis)
            _bravais_vectors = np.array(self.bravais_vectors)
        else:
            _n_cells = None
            _basis = None
            _bravais_vectors = None

        copy = Spin_System(
            self.positions,
            self.spins,
            self.unordered,
            n_cells=_n_cells,
            basis=_basis,
            bravais_vectors=_bravais_vectors,
        )

        return copy

    def deepcopy(self):
        """Make a deep copy of spin system"""

        # Need these field if not unordered
        if not self.unordered:
            _n_cells = np.array(self.n_cells)
            _basis = np.array(self.basis)
            _bravais_vectors = np.array(self.bravais_vectors)
        else:
            _n_cells = None
            _basis = None
            _bravais_vectors = None

        copy = Spin_System(
            np.array(self.positions),
            np.array(self.spins),
            self.unordered,
            n_cells=_n_cells,
            basis=_basis,
            bravais_vectors=_bravais_vectors,
        )

        return copy

    def is_flat(self):
        """Return true if spin_system is flat"""
        return len(self.positions.shape) == 2

    def is_shaped(self):
        """Return true if spin_system is flat"""
        return not self.is_flat()

    def flatten(self):
        """Flatten the spin system"""
        self.positions = np.reshape(self.positions, (self.nos, 3), order="F")
        self.spins = np.reshape(self.spins, (self.nos, 3), order="F")

    def flattened(self):
        """Return a flattend view into the spin system"""
        if self.is_flat():
            return self
        else:
            temp = self.copy()
            temp.positions = np.reshape(self.positions, (self.nos, 3), order="F")
            temp.spins = np.reshape(self.spins, (self.nos, 3), order="F")
            return temp

    @require_order
    def shape(self):
        """Shape the spin system"""

        self.positions = np.reshape(
            self.positions,
            (self.n_cell_atoms, self.n_cells[0], self.n_cells[1], self.n_cells[2], 3),
            order="F",
        )
        self.spins = np.reshape(
            self.spins,
            (self.n_cell_atoms, self.n_cells[0], self.n_cells[1], self.n_cells[2], 3),
            order="F",
        )

    @require_order
    def shaped(self):
        """Return a shaped view into the spin system"""

        if not self.is_flat():
            return self
        else:
            temp = self.copy()
            temp.positions = np.reshape(
                self.positions,
                (
                    self.n_cell_atoms,
                    self.n_cells[0],
                    self.n_cells[1],
                    self.n_cells[2],
                    3,
                ),
                order="F",
            )
            temp.spins = np.reshape(
                self.spins,
                (
                    self.n_cell_atoms,
                    self.n_cells[0],
                    self.n_cells[1],
                    self.n_cells[2],
                    3,
                ),
                order="F",
            )
            return temp

    def center(self):
        return np.mean(self.flattened().positions, axis=0)

    @require_order
    def a_slice(self, val):
        result = self.shaped()[:, val, :, :, :]
        result.n_cells = [1, self.n_cells[1], self.n_cells[2]]
        result.bravais_vectors = self.bravais_vectors
        result.n_cell_atoms = self.n_cell_atoms
        result.unordered = False
        return result.flattened()

    @require_order
    def b_slice(self, val):
        result = self.shaped()[:, :, val, :, :]
        result.n_cells = [self.n_cells[0], 1, self.n_cells[2]]
        result.bravais_vectors = self.bravais_vectors
        result.n_cell_atoms = self.n_cell_atoms
        result.unordered = False
        return result.flattened()

    @require_order
    def c_slice(self, val):
        result = self.shaped()[:, :, :, val, :]
        result.n_cells = [self.n_cells[0], self.n_cells[1], 1]
        result.bravais_vectors = self.bravais_vectors
        result.n_cell_atoms = self.n_cell_atoms
        result.unordered = False
        return result.flattened()


def spin_system_from_p_state(p_state, idx_image=-1):
    from spirit import geometry, system

    # Query information from spirit state
    _positions = geometry.get_positions(p_state, idx_image=idx_image)
    _spins = system.get_spin_directions(p_state, idx_image=idx_image)
    _n_cells = geometry.get_n_cells(p_state)
    _n_cell_atoms = geometry.get_n_cell_atoms(p_state)

    # We need to figure out the bravais vectors and the basis, we will obtain both from the positions array
    _basis = np.array(_positions[:_n_cell_atoms])

    _bravais_vectors = np.zeros(shape=(3, 3))

    if _n_cells[0] > 1:
        _bravais_vectors[0] = _positions[_n_cell_atoms] - _positions[0]

    if _n_cells[1] > 1:
        _bravais_vectors[1] = _positions[_n_cell_atoms * _n_cells[0]] - _positions[0]

    if _n_cells[2] > 1:
        _bravais_vectors[2] = (
            _positions[_n_cell_atoms * _n_cells[0] * _n_cells[1]] - _positions[0]
        )

    spin_system = Spin_System(
        _positions,
        _spins,
        unordered=False,
        n_cells=_n_cells,
        bravais_vectors=_bravais_vectors,
        basis=_basis,
    )

    return spin_system


class energy_path:
    def __init__(self):
        self.reaction_coordinate = []
        self.total_energy = []
        self.energy_contributions = {}
        self.interpolated_reaction_coordinate = []
        self.interpolated_total_energy = []
        self.interpolated_energy_contributions = {}

    def idx_sp(self):
        return np.argmax(self.total_energy)

    def noi(self):
        return len(self.reaction_coordinate)

    def split(self, idx_0, idx_1):
        # TODO: interpolated quantities
        split_path = energy_path()
        split_path.reaction_coordinate = np.array(self.reaction_coordinate[idx_0:idx_1])
        split_path.total_energy = np.array(self.total_energy[idx_0:idx_1])
        return split_path

    def slope(self):
        result = np.gradient(self.total_energy, self.reaction_coordinate)
        return result

    def interpolated_slope(self):
        result = np.gradient(
            self.interpolated_total_energy, self.interpolated_reaction_coordinate
        )
        return result

    def curvature(self):
        result = np.gradient(self.slope(), self.reaction_coordinate)
        return result

    def interpolated_curvature(self):
        result = np.gradient(
            self.interpolated_slope(), self.interpolated_reaction_coordinate
        )
        return result

    def n_interpolated(self):
        return int(
            (len(self.interpolated_reaction_coordinate) - self.noi()) / (self.noi() - 1)
        )

    def barrier(self):
        return np.max(self.total_energy) - self.total_energy[0]


def energy_path_from_p_state(p_state):
    # TODO: contributions
    from spirit import chain

    result = energy_path()
    result.reaction_coordinate = chain.get_reaction_coordinate(p_state)
    result.total_energy = chain.get_energy(p_state)
    result.interpolated_reaction_coordinate = (
        chain.get_reaction_coordinate_interpolated(p_state)
    )
    result.interpolated_total_energy = chain.get_energy_interpolated(p_state)

    if (
        result.interpolated_reaction_coordinate[-1] == 0
    ):  # Quick check if the interpolated quantities have been computed
        result.interpolated_reaction_coordinate = []
        result.interpolated_total_energy = []

    return result
