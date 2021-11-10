from . import util
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_FILE = os.path.join(SCRIPT_DIR, "inputs/test_chain_io.cfg")
OUTPUT_FILE = os.path.join(SCRIPT_DIR, "output_test_chain_io.ovf")

from ..spirit_utils import chain_io
import numpy as np
from spirit import state, chain, configuration, system, io

def test_write_between():
    with state.State(INPUT_FILE) as p_state:
        configuration.plus_z(p_state)
        chain.image_to_clipboard(p_state)

        noi = 10
        chain.set_length(p_state, noi)
        for idx_image in range(noi):
            configuration.skyrmion(p_state, radius=2*idx_image, idx_image=idx_image)

        # Test write between
        idx_1 = 2
        idx_2 = 5

        old_spin_directions = []
        for idx in range(idx_1, idx_2+1):
            old_spin_directions.append( np.array(system.get_spin_directions(p_state, idx_image=idx)) )

        chain_io.chain_write_between(p_state, OUTPUT_FILE, idx_1, idx_2)
        chain.set_length(p_state, 1)
        io.chain_read(p_state, OUTPUT_FILE)

        for i, idx in enumerate(range(idx_1, idx_2 + 1)):
            new_spin_directions = np.array( np.array(system.get_spin_directions(p_state, idx_image=i)) )
            print( "Testing image {}, original idx {}".format(i, idx) )
            print(  np.max( np.ravel(old_spin_directions[i] ) - np.ravel(new_spin_directions ) ) )
            assert( np.max( np.ravel(old_spin_directions[i] ) - np.ravel(new_spin_directions ) ) < 1e-10 )