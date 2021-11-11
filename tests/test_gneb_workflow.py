from . import util
import os
from anytree import Node, NodeMixin, RenderTree 
from anytree.exporter import DotExporter

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

from ..spirit_utils import gneb_workflow, data, plotting
import matplotlib.pyplot as plt
import numpy as np
from spirit import state, chain, configuration, io, transition, simulation

def test_gneb_workflow():
    INPUT_FILE = os.path.join(SCRIPT_DIR, "inputs/test_gneb_workflow.cfg")
    QUIET = True
    OUTPUT_FOLDER = SCRIPT_DIR + "/output_gneb_test"
    INITIAL_CHAIN = SCRIPT_DIR + "/initial_chain_gneb.ovf"
    FINAL_CHAIN   = SCRIPT_DIR + "/final_chain_gneb.ovf"
    NOI = 10

    # First we create the initial chain
    with state.State(INPUT_FILE, QUIET) as p_state:
        configuration.plus_z(p_state)
        configuration.skyrmion(p_state, radius = 5)
        simulation.start(p_state, simulation.METHOD_LLG, simulation.SOLVER_LBFGS_OSO)
        chain.image_to_clipboard(p_state)
        chain.set_length(p_state, NOI)
        configuration.plus_z(p_state, idx_image = NOI-1)
        transition.homogeneous(p_state, 0, NOI-1)
        io.chain_write(p_state, INITIAL_CHAIN)

    gnw = gneb_workflow.GNEB_Node(name="gneb_test", input_file=INPUT_FILE, output_folder=OUTPUT_FOLDER, initial_chain_file=INITIAL_CHAIN)

    gnw.n_iterations_check   = 2000
    gnw.max_total_iterations = 20000
    gnw.convergence          = 1e-3
    gnw.path_shortening_constant = 1e-5

    gnw.setup_plot_callbacks()

    gnw.run()
    gnw.collect_chain(SCRIPT_DIR + "/chain_before_clamp.ovf")
    gnw.clamp_and_refine(max_total_iterations=10000, convergence=1e-3, apply_ci=False)
    gnw.clamp_and_refine(idx_max_list = [2], max_total_iterations=1e4, convergence=1e-5, apply_ci=False)
    gnw.clamp_and_refine(idx_max_list = [2], max_total_iterations=2e4, convergence=1e-8, apply_ci=True, target_noi=3)

    gnw.to_json()
    gnw.collect_chain(FINAL_CHAIN)

    with state.State(INPUT_FILE, QUIET) as p_state:
        io.chain_read(p_state, FINAL_CHAIN)
        path = data.energy_path_from_p_state(p_state)
        plotting.plot_energy_path(path, plt.gca())
        plt.savefig(SCRIPT_DIR + "/final_path.png")
        plt.close()

    with state.State(INPUT_FILE, QUIET) as p_state:
        io.chain_read(p_state, SCRIPT_DIR + "/chain_before_clamp.ovf")
        path = data.energy_path_from_p_state(p_state)
        plotting.plot_energy_path(path, plt.gca())
        plt.savefig(SCRIPT_DIR + "/path_before_clamp.png")

if __name__ == "__main__":
    test_gneb_workflow()