import unittest

from spirit_extras import chain_io
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_FILE = os.path.join(SCRIPT_DIR, "inputs/test_chain_io.cfg")
QUIET = True

import numpy as np
from spirit import state, chain, configuration, system, io


class Chain_IO_Test(unittest.TestCase):
    def setUp(self) -> None:
        pass

    def tearDown(self) -> None:
        pass

    def test_write_between(self):
        OUTPUT_FILE = os.path.join(SCRIPT_DIR, "output_test_chain_io.ovf")
        with state.State(INPUT_FILE, QUIET) as p_state:
            configuration.plus_z(p_state)
            chain.image_to_clipboard(p_state)

            noi = 10
            chain.set_length(p_state, noi)
            for idx_image in range(noi):
                configuration.skyrmion(
                    p_state, radius=2 * idx_image, idx_image=idx_image
                )

            # Test write between
            idx_1 = 2
            idx_2 = 5

            old_spin_directions = []
            for idx in range(idx_1, idx_2 + 1):
                old_spin_directions.append(
                    np.array(system.get_spin_directions(p_state, idx_image=idx))
                )

            chain_io.chain_write_between(p_state, OUTPUT_FILE, idx_1, idx_2)
            chain.set_length(p_state, 1)
            io.chain_read(p_state, OUTPUT_FILE)

            for i, idx in enumerate(range(idx_1, idx_2 + 1)):
                new_spin_directions = np.array(
                    np.array(system.get_spin_directions(p_state, idx_image=i))
                )
                print("Testing image {}, original idx {}".format(i, idx))
                res = np.max(
                    np.ravel(old_spin_directions[i]) - np.ravel(new_spin_directions)
                )
                self.assertAlmostEqual(res, 1e-10)

    def test_write_split_at(self):
        with state.State(INPUT_FILE, QUIET) as p_state:
            configuration.plus_z(p_state)
            chain.image_to_clipboard(p_state)

            noi = 10
            chain.set_length(p_state, noi)
            for idx_image in range(noi):
                configuration.skyrmion(
                    p_state, radius=2 * idx_image, idx_image=idx_image
                )

            # Test write split_at
            filenames = [
                os.path.join(SCRIPT_DIR, n) for n in ["output_1.ovf", "output_2.ovf"]
            ]
            idx_list = [2, 3, 9]
            idx_pairs = [[2, 3], [3, 9]]

            old_spin_directions = []
            for p in idx_pairs:
                old_spin_directions.append([])
                for idx in range(p[0], p[1] + 1):
                    old_spin_directions[-1].append(
                        np.array(system.get_spin_directions(p_state, idx_image=idx))
                    )

            chain_io.chain_write_split_at(
                p_state, idx_list=idx_list, filename_list=filenames
            )

            for j, (f, p) in enumerate(zip(filenames, idx_pairs)):
                chain.set_length(p_state, 1)
                io.chain_read(p_state, f)
                idx_1 = p[0]
                idx_2 = p[1]
                for i, idx in enumerate(range(idx_1, idx_2 + 1)):
                    new_spin_directions = np.array(
                        np.array(system.get_spin_directions(p_state, idx_image=i))
                    )
                    print("Testing image {}, original idx {}".format(i, idx))
                    res = np.max(
                        np.ravel(old_spin_directions[j][i])
                        - np.ravel(new_spin_directions)
                    )
                    self.assertAlmostEqual(res, 1e-10)

    def test_append_to_file_from_file(self):
        FILE_IN = os.path.join(SCRIPT_DIR, "in.ovf")
        FILE_OUT = os.path.join(SCRIPT_DIR, "out.ovf")

        if os.path.exists(FILE_OUT):
            os.remove(FILE_OUT)

        with state.State(INPUT_FILE, QUIET) as p_state:
            configuration.plus_z(p_state)
            chain.image_to_clipboard(p_state)

            noi = 2
            chain.set_length(p_state, noi)
            for idx_image in range(noi):
                configuration.skyrmion(
                    p_state, radius=2 * idx_image, idx_image=idx_image
                )

            old_spin_directions = []
            for idx in range(noi):
                old_spin_directions.append(
                    np.array(system.get_spin_directions(p_state, idx_image=idx))
                )

            io.chain_write(p_state, FILE_IN)

            chain.set_length(p_state, 1)
            configuration.random(p_state)

            chain_io.chain_append_to_file_from_file(p_state, FILE_OUT, FILE_IN)
            chain_io.chain_append_to_file_from_file(p_state, FILE_OUT, FILE_IN)
            chain_io.chain_append_to_file_from_file(p_state, FILE_OUT, FILE_IN)
            chain_io.chain_append_to_file_from_file(p_state, FILE_OUT, FILE_IN)

            chain.set_length(p_state, 1)
            io.chain_read(p_state, FILE_OUT)

            for i in range(4):
                print("Testing image {}, original idx {}".format(i, idx))

                new_spin_directions = np.array(
                    np.array(system.get_spin_directions(p_state, idx_image=2 * i))
                )
                res = np.max(
                    np.ravel(old_spin_directions[0]) - np.ravel(new_spin_directions)
                )
                self.assertAlmostEqual(res, 1e-10)

                new_spin_directions = np.array(
                    np.array(system.get_spin_directions(p_state, idx_image=2 * i + 1))
                )
                res = np.max(
                    np.ravel(old_spin_directions[1]) - np.ravel(new_spin_directions)
                )
                self.assertAlmostEqual(res, 1e-10)
