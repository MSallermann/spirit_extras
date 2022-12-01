import unittest
import os, shutil
from spirit_extras import data
import numpy as np

SCRIPT_DIR = os.path.dirname(__file__)


class Data_Test(unittest.TestCase):
    def setUp(self) -> None:
        pass

    def tearDown(self) -> None:
        pass

    def test_create_self(self):
        nos = 10
        positions = np.array([[i * 1.05, i / 0.99, -i * 0.75] for i in range(nos)])
        spins = np.array([[0, 0, 1.0] for _ in range(nos)])

        system = data.Spin_System(positions, spins)

        # Some basic properties
        self.assertEqual(system.nos, nos)
        self.assertTrue(np.all(system.positions[4] == positions[4]))
        self.assertTrue(np.all(system.spins[4] == spins[4]))
        self.assertTrue(np.all(system.center() == np.mean(positions, axis=0)))

        # Slicing
        for sl in [
            slice(-6, 10, 2),
            slice(10, 1, -1),
            slice(5, 5, 1),
            slice(5, 6),
            system.positions[:, 0] > 3,
        ]:
            system_slice = system[sl]
            self.assertEqual(system_slice.nos, len(positions[sl]))
            self.assertTrue(np.all(system_slice.positions == positions[sl]))
            self.assertTrue(np.all(system_slice.spins == spins[sl]))

        # Assigning to the fields in spin system should change the underlying buffers
        system.spins[2] = np.array([1, 1, 1])
        self.assertTrue(np.all(spins[2] == np.array([1, 1, 1])))

        # The same should be true for a shallow copy
        system_copy = system.copy()
        system_copy.spins[3] = np.array([1, 1, 1])
        self.assertTrue(np.all(spins[3] == np.array([1, 1, 1])))

        # Assigning to a deepcopy should not change anythin in the original buffer
        system_copy = system.deepcopy()
        system_copy.spins[4] = np.array([1, 1, 1])
        self.assertTrue(np.all(spins[4] == np.array([0, 0, 1.0])))

        # Assigning to a slice should also work
        mask1 = [i < 5 for i in range(nos)]
        mask2 = [i >= 5 for i in range(nos)]
        system.spins[mask1] = [0, 0, 2]
        system.spins[mask2] = [0, 0, -2]
        self.assertTrue(np.all(spins[mask1] == [0, 0, 2]))
        self.assertTrue(np.all(spins[mask2] == [0, 0, -2]))

        # Misc
        self.assertTrue(system.is_flat())
        self.assertFalse(system.is_shaped())

        # Should at least not throw anything
        system.flatten()
        system.flattened()

    def test_from_p_state(self):
        INPUT_FILE = os.path.join(SCRIPT_DIR, "inputs/test_data.cfg")

        # These are the data values which are hardcoded in the INPUT_FILE

        lattice_constant_input = 1.5
        bravais_input = np.array([[0.9, -0.5, 0.0], [-0.1, 0.8, 0.0], [0.0, 0.2, 2.0]])
        basis_input = np.array([[0.0, 0.2, 0.0], [0.5, 0.5, 0.5], [0.1, 0.8, 0.1]])
        n_cells_input = [20, 2, 17]

        # CAREFUL: For our test we have to multiply the bravais vectors by the lattice constant and transform the basis from bravais coordiantes to real space coordinates
        bravais_check = lattice_constant_input * bravais_input
        basis_check = np.matmul(basis_input, bravais_check)

        from spirit import state, configuration, system

        with state.State(INPUT_FILE, quiet=True) as p_state:

            spins = system.get_spin_directions(p_state)

            configuration.plus_z(p_state)  # set +z configuration
            spin_system = data.spin_system_from_p_state(p_state)

            # Check for n_cells, n_cell_atoms and nos
            self.assertEqual(spin_system.nos, np.prod(n_cells_input) * len(basis_input))
            self.assertEqual(spin_system.n_cell_atoms, len(basis_input))
            self.assertTrue(np.all(spin_system.n_cells == n_cells_input))

            # Check basis and bravais vectors
            self.assertTrue(
                np.allclose(spin_system.basis.flatten(), basis_check.flatten())
            )
            self.assertTrue(
                np.allclose(
                    spin_system.bravais_vectors.flatten(), bravais_check.flatten()
                )
            )

            # Test assigning to a system slice (DOES NOT WORK WITH FANCY INDEXING)
            system_slice = spin_system[::2]  # Flip spins in a spatial region
            system_slice.spins[:] = np.ones(shape=(system_slice.nos, 3))
            self.assertTrue(np.all(spins[::2] == np.ones(shape=(system_slice.nos, 3))))

            # Test index and tupel
            ib, a, b, c = 1, 3, 0, 9  # test for consistency in the middle
            idx = spin_system.idx(ib, a, b, c)
            tupel = spin_system.tupel(idx)
            idx2 = spin_system.idx(*tupel)
            self.assertEqual(idx, idx2)
            self.assertEqual([ib, a, b, c], tupel)

            tupel_end = [
                len(basis_input) - 1,
                n_cells_input[0] - 1,
                n_cells_input[1] - 1,
                n_cells_input[2] - 1,
            ]  # test for correctness at the end point
            idx_end = spin_system.nos - 1
            self.assertEqual(spin_system.idx(*tupel_end), idx_end)
            self.assertEqual(spin_system.tupel(idx_end), tupel_end)

            # Shape etc.
            self.assertTrue(spin_system.is_flat())
            self.assertFalse(spin_system.is_shaped())

            # Should not do anything
            spin_system.flatten()
            spin_system.flattened()

            # Test shaped view
            spin_system_shaped_view = spin_system.shaped()
            self.assertFalse(spin_system_shaped_view.is_flat())
            self.assertTrue(spin_system_shaped_view.is_shaped())
            self.assertEqual(
                spin_system_shaped_view.positions.shape,
                (len(basis_input), *n_cells_input, 3),
            )
            self.assertEqual(
                spin_system_shaped_view.spins.shape,
                (len(basis_input), *n_cells_input, 3),
            )
            self.assertTrue(spin_system.shaped())  # old system should remain flat

            # Assignment should work
            spin_system_shaped_view.spins[0, 3, 1, 3] = [-2, -2, -2]
            idx = spin_system_shaped_view.idx(0, 3, 1, 3)
            self.assertTrue(np.all(spins[idx] == [-2, -2, -2]))

            # Shape the spin system
            spin_system.shape()
            self.assertTrue(spin_system.is_shaped())
            spin_system.spins[0, 3, 1, 3] = [-3, -3, -3]
            idx = spin_system.idx(0, 3, 1, 3)
            self.assertTrue(np.all(spins[idx] == [-3, -3, -3]))

            print(spin_system.center())
            print(spin_system.flattened().center())


if __name__ == "__main__":
    t = Data_Test()
    t.test_create_self()
    t.test_from_p_state()
