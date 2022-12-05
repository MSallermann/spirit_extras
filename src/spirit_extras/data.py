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
        n_cell_atoms=None,
    ):
        self.spins = np.asarray(spins, dtype=float)
        self._basis = basis
        self._unordered = unordered
        self._n_cells = n_cells
        self._bravais_vectors = bravais_vectors

        if positions is None:
            self._positions = np.zeros(
                shape=self.spins.shape
            )  # Sometimes the positions are not of interest, so we provide the option not to specify them
        else:
            self._positions = np.asarray(positions, dtype=float)

        if basis is None:
            if not n_cell_atoms is None:
                self._basis = np.zeros(
                    shape=(n_cell_atoms, 3)
                )  # Same thing as above with positions

        self.check_shape_of_spins_positions()

        if not self.unordered:
            self.checks_for_ordered_system()

    def check_shape_of_spins_positions(self):
        # Should have the same shape
        if self.positions.shape != self.spins.shape:
            raise Exception(
                "Positions and spins have different shapes ... {} and {} respectively".format(
                    self.positions.shape, self.spins.shape
                )
            )

        # Make sure the shape is of valid format
        if (
            not (len(self.positions.shape) == 2 or len(self.positions.shape) == 5)
            and self.positions.shape[-1] == 3
        ):
            raise Exception(
                "Spins and positions must either have shape (nos,3) or (n_cell_atoms, n_cells[0], n_cells[1], n_cells[2], 3), but they have shape {}".format(
                    self.positions.shape
                )
            )

        # If shaped positions and spins are given, we perform some consistency checks and try to infer n_cells, if it was not given
        if self.is_shaped():
            if self._unordered:
                raise Exception(
                    "Only arrays of shape (nos,3) can be used when unordered=True"
                )

            if not self._n_cells is None:
                if self.positions.shape[1:-1] != (
                    self.n_cells[0],
                    self.n_cells[1],
                    self.n_cells[2],
                ):
                    raise Exception(
                        "Shape of positions/spins {} does not match n_cells {}".format(
                            self.positions.shape, self._n_cells
                        )
                    )
            else:
                self._n_cells = np.array(self.positions.shape[1:-1])

            if not self._basis is None:
                if self._positions.shape[0] != len(self._basis):
                    raise Exception(
                        "Shape of positions/spins {} does not match length of basis {}".format(
                            self.positions.shape, len(self.basis)
                        )
                    )
            else:
                self._basis = np.zeros(
                    shape=(self.positions.shape[0], 3)
                )  # Dummy basis

    def checks_for_ordered_system(self):
        if self.n_cells is None:
            raise Exception("For an ordered system 'n_cells' must be provided")
        else:
            self._n_cells = np.asarray(self.n_cells, dtype=int)

        if self._n_cells.shape != (3,):
            raise Exception(
                "n_cells has wrong shape. It should be (3), but is {}".format(
                    self._n_cells.shape
                )
            )

        # basis
        if self._basis is None:
            self._basis = np.array([[0, 0, 0]])  # default for single basis systems
        else:
            self._basis = np.asarray(self._basis, dtype=float)
        if not len(self.basis.shape) == 2 or not self.basis.shape[-1] == 3:
            raise Exception(
                "`basis` has wrong shape. It should have shape (n_cell_atoms, 3), but has shape {}".format(
                    self.basis.shape
                )
            )

        # bravais vectors
        if self._bravais_vectors is None:
            self._bravais_vectors = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        else:
            self._bravais_vectors = np.asarray(self._bravais_vectors, dtype=float)
        if not self.bravais_vectors.shape == (3, 3):
            raise Exception(
                "`bravais_vectors` has wrong shape. It should have shape (3, 3), but has shape {}".format(
                    self.bravais_vectors.shape
                )
            )

        # nos
        nos_expected = np.prod(self.n_cells) * self.n_cell_atoms
        if nos_expected != self.nos:
            raise Exception(
                "NOS from n_cells/n_cell_atoms ({}) does not match spin arrays ({})".format(
                    nos_expected, self.nos
                )
            )

    def require_order(func):
        from functools import wraps

        @wraps(func)
        def wrapper(self, *args, **kw):
            if self._unordered:
                raise Exception("This function requires on ordered system!")
            return func(self, *args, **kw)

        return wrapper

    def __getitem__(self, key):
        sliced_spin_system = Spin_System(self._positions[key], self.spins[key])
        return sliced_spin_system

    def __str__(self):
        result = f"nos       = {self.nos}\n"
        result += f"unordered = {self.unordered}\n"
        result += f"flat      = {self.is_flat()}\n"
        result += (
            f"positions = {type(self.positions)}, shape = {self.positions.shape}\n"
        )
        result += f"spins     = {type(self.spins)}, shape = {self.spins.shape}\n"

        if not self.unordered:
            result += f"n_cells         = {self.n_cells}\n"
            result += f"n_cell_atoms    = {self.n_cell_atoms}\n"
            result += f"bravais_vectors:\n{self.bravais_vectors}\n"
            result += f"basis:\n{self.basis}\n"

        return result

    @property
    def positions(self):
        return self._positions

    @property
    @require_order
    def n_cell_atoms(self):
        return len(self._basis)

    @property
    @require_order
    def n_cells(self):
        return self._n_cells

    @property
    @require_order
    def bravais_vectors(self):
        return self._bravais_vectors

    @property
    @require_order
    def basis(self):
        return self._basis

    @property
    def nos(self):
        if self.is_flat():
            return len(self.spins)
        else:
            return (
                self.n_cells[0] * self.n_cells[1] * self.n_cells[2] * self.n_cell_atoms
            )

    @property
    def unordered(self):
        return self._unordered

    @require_order
    def idx(self, ib, a, b, c, periodic=False):
        """Computes the linear idx from coords in the bravais lattice.

        Args:
            ib (int): idx of spin in basis cell
            a (int): first bravais translation index
            b (int): second bravais translation index
            c (int): third bravais translation index

        Returns:
            int: the linear index for the flattened spin/position arrays
        """

        def check_bounds(var, name, lower, upper):
            if var >= upper or var < 0:
                raise Exception(
                    "`{}` is {}, but has to lie within {} to {}".format(
                        name, var, lower, upper - 1
                    )
                )

        if not periodic:
            check_bounds(ib, "ib", 0, self.n_cell_atoms)
            check_bounds(a, "a", 0, self.n_cells[0])
            check_bounds(b, "b", 0, self.n_cells[1])
            check_bounds(c, "c", 0, self.n_cells[2])
        else:
            # Unlike C or C++, pythons modulo operator always returns numbers with the same sign as the denominator
            ib = ib % self.n_cell_atoms
            a = a % self.n_cells[0]
            b = b % self.n_cells[1]
            c = c % self.n_cells[2]

        return int(
            ib + self.n_cell_atoms * (a + self._n_cells[0] * (b + self._n_cells[1] * c))
        )

    @require_order
    def tupel(self, idx):
        """Computes the tupel of bravais indices from a linear idx.

        Args:
            idx (int): linear index

        Returns:
            list(int,int,int,int): list of bravais indices [idx_cell_atom, a, b, c]
        """
        if idx < 0 or idx >= self.nos:
            raise Exception(
                "`idx is {}, but has to lie within 0 and {}`".format(idx, self.nos - 1)
            )

        idx_diff = idx

        maxVal = np.array([self.n_cell_atoms, *self._n_cells])
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
        if not self._unordered:
            _n_cells = np.array(self._n_cells)
            _basis = np.array(self._basis)
            _bravais_vectors = np.array(self._bravais_vectors)
        else:
            _n_cells = None
            _basis = None
            _bravais_vectors = None

        copy = Spin_System(
            self._positions,
            self.spins,
            self._unordered,
            n_cells=_n_cells,
            basis=_basis,
            bravais_vectors=_bravais_vectors,
        )

        return copy

    def deepcopy(self):
        """Make a deep copy of spin system"""

        # Need these field if not unordered
        if not self._unordered:
            _n_cells = np.array(self._n_cells)
            _basis = np.array(self._basis)
            _bravais_vectors = np.array(self._bravais_vectors)
        else:
            _n_cells = None
            _basis = None
            _bravais_vectors = None

        copy = Spin_System(
            np.array(self._positions),
            np.array(self.spins),
            self._unordered,
            n_cells=_n_cells,
            basis=_basis,
            bravais_vectors=_bravais_vectors,
        )

        return copy

    def is_flat(self):
        """Return true if spin_system is flat"""
        return len(self._positions.shape) == 2

    def is_shaped(self):
        """Return true if spin_system is flat"""
        return not self.is_flat()

    def flatten(self):
        """Flatten the spin system"""
        self._positions = np.reshape(self._positions, (self.nos, 3), order="F")
        self.spins = np.reshape(self.spins, (self.nos, 3), order="F")

    def flattened(self):
        """Return a flattend view into the spin system"""
        if self.is_flat():
            return self
        else:
            temp = self.copy()
            temp._positions = np.reshape(self._positions, (self.nos, 3), order="F")
            temp.spins = np.reshape(self.spins, (self.nos, 3), order="F")
            return temp

    @require_order
    def shape(self):
        """Shape the spin system"""

        self._positions = np.reshape(
            self._positions,
            (
                self.n_cell_atoms,
                self.n_cells[0],
                self.n_cells[1],
                self.n_cells[2],
                3,
            ),
            order="F",
        )
        self.spins = np.reshape(
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

    @require_order
    def shaped(self):
        """Return a shaped view into the spin system"""

        if not self.is_flat():
            return self
        else:
            temp = self.copy()
            temp._positions = np.reshape(
                self._positions,
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
        result.bravais_vectors = self._bravais_vectors
        result.n_cell_atoms = self.n_cell_atoms
        result.unordered = False
        return result.flattened()

    @require_order
    def b_slice(self, val):
        result = self.shaped()[:, :, val, :, :]
        result.n_cells = [self.n_cells[0], 1, self.n_cells[2]]
        result.bravais_vectors = self._bravais_vectors
        result.n_cell_atoms = self.n_cell_atoms
        result.unordered = False
        return result.flattened()

    @require_order
    def c_slice(self, val):
        result = self.shaped()[:, :, :, val, :]
        result.n_cells = [self.n_cells[0], self.n_cells[1], 1]
        result.bravais_vectors = self._bravais_vectors
        result.n_cell_atoms = self.n_cell_atoms
        result.unordered = False
        return result.flattened()


def infer_lattice(n_cells, n_cell_atoms, positions):
    """Infers the bravais_vectors and the bravais basis from an array of positions, n_cells and n_cell_atoms.

    Args:
        n_cells (int[3]): number of bravais cells in a/b/c direction
        n_cell_atoms (int): number of spins per bravais cell
        positions (float[nos,3]): linear array of positions

    Raises:
        Exception: if length of positions does not match

    Returns:
        float[3,3], float[n_cell_atoms, 3]: tupel of bravais vectors in direction a,b,c and basis vectors
    """

    nos_expected = np.prod(n_cells) * n_cell_atoms
    nos = len(positions)
    if nos != nos_expected:
        raise Exception(
            "Length of positions array ({}) does not match number of expected spins ({}).".format(
                nos, nos_expected
            )
        )

    basis = np.array(positions[:n_cell_atoms])

    bravais_vectors = np.zeros(shape=(3, 3))

    if n_cells[0] > 1:
        bravais_vectors[0] = positions[n_cell_atoms] - positions[0]

    if n_cells[1] > 1:
        bravais_vectors[1] = positions[n_cell_atoms * n_cells[0]] - positions[0]

    if n_cells[2] > 1:
        bravais_vectors[2] = (
            positions[n_cell_atoms * n_cells[0] * n_cells[1]] - positions[0]
        )

    return bravais_vectors, basis


def spin_system_from_p_state(p_state, idx_image=-1):
    from spirit import geometry, system

    # Query information from spirit state
    _positions = geometry.get_positions(p_state, idx_image=idx_image)
    _spins = system.get_spin_directions(p_state, idx_image=idx_image)
    n_cells = geometry.get_n_cells(p_state)
    n_cell_atoms = geometry.get_n_cell_atoms(p_state)

    _bravais_vectors, _basis = infer_lattice(n_cells, n_cell_atoms, _positions)

    spin_system = Spin_System(
        _positions,
        _spins,
        unordered=False,
        n_cells=n_cells,
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
