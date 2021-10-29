from anytree import Node, NodeMixin, RenderTree 
from anytree.exporter import DotExporter
import os

import shutil
import copy

import traceback

from numpy import inf

from .data import energy_path, energy_path_from_p_state
from .util import set_output_folder
from .chain_io import chain_write_split_at
from datetime import datetime

class GNEB_Node(NodeMixin):
    """A class that represents a GNEB calculation on a single chain. Can spawn children if cutting of the chain becomes necessary."""

    chain_file : str             = ""
    initial_chain_file : str     = ""
    input_file : str             = ""
    gneb_workflow_log_file : str = ""
    current_energy_path = object()
    n_iterations_check  = 5000
    n_checks_save       = 3
    total_iterations    = 0
    intermediate_minima = []

    target_noi  = 10
    convergence = 1e-5

    state_prepare_callback = None
    gneb_step_callback     = None
    exit_callback          = None
    before_gneb_callback   = None
    before_llg_callback    = None

    output_folder = ""
    output_tag    = ""

    _converged    = False

    def __init__(self, name, input_file, output_folder, initial_chain_file=None, gneb_workflow_log_file=None, parent=None, children=None):

        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
        self.output_folder = output_folder

        self.chain_file = output_folder + "/chain.ovf"

        if(initial_chain_file):
            if not os.path.exists(initial_chain_file):
                raise Exception("Initial chain file does not exist!")
            self.initial_chain_file = output_folder + "/root_initial_chain.ovf"
            shutil.copyfile(initial_chain_file, self.initial_chain_file)
            shutil.copyfile(initial_chain_file, self.chain_file)

        self.input_file = input_file
        self.name = name
        self.parent = parent
        if not gneb_workflow_log_file:
            self.gneb_workflow_log_file = self.output_folder + "/workflow_log.txt"
        else:
            self.gneb_workflow_log_file = gneb_workflow_log_file
        if children:
            self.children = children

        creation_msg = "Creating new GNEB_Node '{}'".format(name)
        if(parent):
            creation_msg += ", parent '{}'".format(parent.name)
        if(children):
            creation_msg += ", children: "
            for c in children:
                creation_msg += "{} ".format(c.name)
        self.log(creation_msg)

    def log(self, message):
        """Append a message with date/time information to the log file."""
        now = datetime.now()
        current_time = now.strftime("%m/%d/%Y, %H:%M:%S")
        log_string = "{} [{:^35}] : {}".format(current_time, self.name, message)
        with open(self.gneb_workflow_log_file, "a") as f:
            print(log_string, file=f)

    def update_energy_path(self, p_state):
        """Updates the current energy path."""
        self.current_energy_path = energy_path_from_p_state(p_state)

    def check_for_minima(self, tol=0.05):
        """
           Checks the chain for intermediate minima. Tol controls how sensitive this check is, lower tol means more sensitive. 
           E.g a tol of 0.1 means that the minimum energy difference between the image in question and any 
           of the two neighbouring images has to be larger than 10% of the images energy to count as an intermediate minimumg.
           Default tol = 0.05
        """
        self.intermediate_minima = []
        e = self.current_energy_path.total_energy
        for i in range(1, len(e) - 1): # Leave out the first and the last energy
            if(e[i-1] > e[i] and e[i+1] > e[i] ):
                
                if(min(e[i-1] - e[i], e[i+1] - e[i]) > abs(tol) * e[i] ):
                    self.intermediate_minima.append(i)

        if(len(self.intermediate_minima) > 0):
            self.log("Found intermediate minima at: {}".format(self.intermediate_minima))

    def save_chain(self, p_state):
        """Saves the chain and overwrites the chain_file"""
        from spirit import io
        self.log("Writing chain to {}".format(self.chain_file))
        io.chain_write(p_state, self.chain_file)

    def spawn_children(self, p_state):
        """Creates child nodes"""
        if self._converged:
            self.log("Converged!")
            return

        self.log("Spawning children")

        from spirit import chain

        # Instantiate the GNEB nodes
        noi = chain.get_noi(p_state)

        idx_list = [0, *self.intermediate_minima, noi-1]

        idx_pair_list = [ (idx_list[i], idx_list[i+1]) for i in range(len(idx_list)-1) ]

        for i1,i2 in idx_pair_list:
            # Attributes that change due to tree structure
            child_name          = self.name + "_{}".format(len(self.children))
            child_input_file    = self.input_file
            child_output_folder = self.output_folder + "/{}".format(len(self.children))
            self.children      += (GNEB_Node(name = child_name, input_file = child_input_file, output_folder = child_output_folder, gneb_workflow_log_file=self.gneb_workflow_log_file, parent = self), )

            self.children[-1].current_energy_path = self.current_energy_path.split(i1, i2+1)

            # Copy the other attributes
            self.children[-1].target_noi             = self.target_noi
            self.children[-1].convergence            = self.convergence
            self.children[-1].state_prepare_callback = self.state_prepare_callback
            self.children[-1].gneb_step_callback     = self.gneb_step_callback
            self.children[-1].exit_callback          = self.exit_callback
            self.children[-1].before_gneb_callback   = self.before_gneb_callback
            self.children[-1].before_llg_callback    = self.before_llg_callback
            self.children[-1].n_iterations_check     = self.n_iterations_check
            self.children[-1].n_checks_save          = self.n_checks_save

        chain_filename_list = [c.chain_file for c in self.children]
        chain_write_split_at(p_state, filename_list=chain_filename_list, idx_list=idx_list)


    def run_children(self, p_state):
        # The following list determines the order in which we run the children of this node.
        # We sort the run from largest to smallest energy barrier (hence the minus).
        # We do this to explore the most 'interesting' paths first

        idx_children_run_order = list(range(len(self.children)))
        idx_children_run_order.sort(key = lambda i : -self.children[i].current_energy_path.barrier())

        for i in idx_children_run_order:
            self.children[i].run()


    def chain_rebalance(self, p_state, tol=0.25):
        """Tries to rebalance the chain while keeping the number of images constant. The closer tol is to zero, the more aggressive the rebalancing."""
        import numpy as np
        from spirit import chain, transition
        from spirit.parameters import gneb
        noi = chain.get_noi(p_state)

        idx_max  = np.argmax(self.current_energy_path.total_energy)

        delta_Rx   = [ self.current_energy_path.reaction_coordinate[i+1] - self.current_energy_path.reaction_coordinate[i] for i in range(noi-1) ]
        delta_Rx_2 = [ self.current_energy_path.reaction_coordinate[i+2] - self.current_energy_path.reaction_coordinate[i] for i in range(noi-2) ]
        max_delta_Rx  = np.max(delta_Rx)
        min_delta_Rx2 = np.min(delta_Rx_2)

        if max_delta_Rx > (1 + np.abs(tol)) * min_delta_Rx2:
            self.log("Rebalancing chain")
            self.log("      Max. Delta Rx {}, Min. Delta Rx2 {}".format(max_delta_Rx, min_delta_Rx2))
            idx   = np.argmax(delta_Rx)
            idx_2 = np.argmin(delta_Rx_2)
            self.log("      Inserting after idx {}, deleting idx {}".format(idx, idx_2+1))

            # Delete the image that was too densely spaced. this will shift all indices greater than idx_2+1
            chain.delete_image(p_state, idx_image = idx_2+1)

            # Correct the index if necessary
            if(idx > idx_2+1) < idx:
                idx -= 1

            chain.insert_image_after(p_state, idx)
            transition.homogeneous(p_state, idx, idx+2)

    def increase_noi(self, p_state, target_noi=None):
        """Increases the noi by (roughly) a factor of two until the number of images is at least as large as target_noi"""
        from spirit import chain, transition

        if not target_noi:
            target_noi = self.target_noi

        noi = chain.get_noi(p_state)

        if(noi <= self.target_noi):
            self.log("Too few images ({}). Inserting additional interpolated images".format(noi))

        while(noi <= self.target_noi):
            transition.homogeneous_insert_interpolated(p_state, 1)
            noi = chain.get_noi(p_state)

        self.log("Number of images = {}".format(noi))

    def check_run_condition(self):
        """Returns True if the run loop should be continued."""
        return (not self._converged) and (len(self.intermediate_minima) <= 0)


    def run(self):
        """Run GNEB with checks after a certain number of iterations"""

        try:
            self.log("Running")

            from spirit import state, simulation, io, transition, chain

            with state.State(self.input_file) as p_state:
                set_output_folder(p_state, self.output_folder, self.output_tag)

                if self.state_prepare_callback:
                    self.state_prepare_callback(self, p_state)

                if not os.path.exists(self.chain_file):
                    raise Exception("Chain file does not exist!")

                io.chain_read(p_state, self.chain_file)
                self.increase_noi(p_state)
                self.update_energy_path(p_state)

                if self.before_gneb_callback:
                    self.before_gneb_callback(self, p_state)

                try:
                    n_checks = 0
                    while(self.check_run_condition()):

                        info = simulation.start(p_state, simulation.METHOD_GNEB, simulation.SOLVER_VP_OSO, n_iterations=self.n_iterations_check)
                        self.update_energy_path(p_state)
                        self.check_for_minima()
                        n_checks += 1
                        self.total_iterations += info.total_iterations

                        self.log("Total iterations = {}".format(self.total_iterations))
                        self.log("      max.torque = {}".format(info.max_torque))
                        self.log("      ips        = {}".format(info.total_ips))
                        self.log("      Delta E    = {}".format(self.current_energy_path.barrier()))

                        self._converged = info.max_torque < self.convergence

                        if(self.gneb_step_callback):
                            self.gneb_step_callback(self, p_state)
                        if(n_checks % self.n_checks_save == 0):
                            self.save_chain(p_state)

                        self.chain_rebalance(p_state)

                except KeyboardInterrupt as e:
                    self.log("Interrupt during run loop")
                    self.save_chain(p_state)
                    if(self.exit_callback):
                        self.exit_callback(self, p_state)
                    raise e

                if self.before_llg_callback:
                    self.before_llg_callback(self, p_state)

                self.log("Relaxing intermediate minima")
                for idx_minimum in self.intermediate_minima:
                    simulation.start(p_state, simulation.METHOD_LLG, simulation.SOLVER_LBFGS_OSO, idx_image = idx_minimum)

                self.update_energy_path(p_state)

                if(self.exit_callback):
                    self.exit_callback(self, p_state)

                self.save_chain(p_state)

                self.spawn_children(p_state)
                self.run_children(p_state)
                # p_state gets deleted here

            self.log("Finished!")

        except Exception as e:
            self.log("Exception during 'run': {}".format(str(e))) # Log the exception and re-raise
            self.log(traceback.format_exc())
            raise e