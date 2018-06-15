from spirit_utils import spirit_path
spirit_path.find_spirit("~/Coding/spirit")

from spirit_utils import plotting
from spirit import state, configuration

import matplotlib.pyplot as plt

fig, ax = plt.subplots()

with state.State("test/input.cfg", quiet=True) as p_state:
    configuration.plus_z(p_state)
    configuration.skyrmion(p_state, 10)

    # plotting.plot_basis_cell(p_state, ax)
    plotting.plot_spins(p_state, ax)

    plt.show()