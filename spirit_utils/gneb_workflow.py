from anytree import Node, NodeMixin, RenderTree 
from anytree.exporter import DotExporter
import os

import shutil
import copy

import numpy as np
import traceback

from numpy import inf

from .data import energy_path, energy_path_from_p_state
from .util import set_output_folder
from .chain_io import chain_write_between, chain_write_split_at
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
    noi = -1
    convergence = 1e-4
    max_total_iterations = -1

    output_folder = ""
    output_tag    = ""

    child_indices = []

    allow_split = True
    state_prepare_callback = None
    gneb_step_callback     = None
    exit_callback          = None
    before_gneb_callback   = None
    before_llg_callback    = None

    _converged    = False

    def __init__(self, name, input_file, output_folder, initial_chain_file=None, gneb_workflow_log_file=None, parent=None, children=None):
        """Constructor."""

        # Create output folder
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
        self.output_folder = output_folder

        # The current chain is always saved here
        self.chain_file = output_folder + "/chain.ovf"

        # If an initial chain is specified we copy it to the output folder
        if(initial_chain_file):
            if not os.path.exists(initial_chain_file):
                raise Exception("Initial chain file does not exist!")
            self.initial_chain_file = output_folder + "/root_initial_chain.ovf"
            shutil.copyfile(initial_chain_file, self.initial_chain_file)
            shutil.copyfile(initial_chain_file, self.chain_file)

        self.input_file = input_file
        self.name       = name
        self.parent     = parent

        # If no log file has been specified we put one in the output folder
        if not gneb_workflow_log_file:
            self.gneb_workflow_log_file = self.output_folder + "/workflow_log.txt"
        else:
            self.gneb_workflow_log_file = gneb_workflow_log_file

        if children:
            self.children = children

        # Log the node creation
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

    def update_energy_path(self, p_state=None):
        """Updates the current energy path. If p_state is given we just use that, otherwise we have to construct it first"""
        if p_state:
        self.current_energy_path = energy_path_from_p_state(p_state)
        else:
            from spirit import state, io
            with state.State(self.input_file) as p_state:
                # Set the output folder for the files created by spirit
                set_output_folder(p_state, self.output_folder, self.output_tag)

                # The state prepare callback can be used to change the state before execution of any other commands
                # One could e.g. use the hamiltonian API to change interaction parameters instead of relying only on the input file
                if self.state_prepare_callback:
                    self.state_prepare_callback(self, p_state)

                # Before we run we must make sure that the chain.ovf file exists now
                if not os.path.exists(self.chain_file):
                    raise Exception("Chain file does not exist!")

                # Read the file, increase up to at leas target_noi and update the energy_path (for plotting etc.)
                io.chain_read(p_state, self.chain_file)
                self.increase_noi(p_state)
                self.update_energy_path(p_state)


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
                
                if(abs(min(e[i-1] - e[i], e[i+1] - e[i])) > abs(tol * e[i]) ):
                    self.intermediate_minima.append(i)

        if(len(self.intermediate_minima) > 0):
            self.log("Found intermediate minima at: {}".format(self.intermediate_minima))

    def save_chain(self, p_state):
        """Saves the chain and overwrites the chain_file"""
        from spirit import io
        self.log("Writing chain to {}".format(self.chain_file))
        io.chain_write(p_state, self.chain_file)

    def add_child(self, p_state, i1, i2):
        # Attributes that change due to tree structure
        child_name          = self.name + "_{}".format(len(self.children))
        child_input_file    = self.input_file
        child_output_folder = self.output_folder + "/{}".format(len(self.children))
        self.children      += (GNEB_Node(name = child_name, input_file = child_input_file, output_folder = child_output_folder, gneb_workflow_log_file=self.gneb_workflow_log_file, parent = self), )

        self.children[-1].current_energy_path = self.current_energy_path.split(i1, i2+1)

        # Copy the other attributes
        self.children[-1].target_noi             = self.target_noi
        self.children[-1].convergence            = self.convergence
        self.children[-1].max_total_iterations   = self.max_total_iterations
        self.children[-1].state_prepare_callback = self.state_prepare_callback
        self.children[-1].gneb_step_callback     = self.gneb_step_callback
        self.children[-1].exit_callback          = self.exit_callback
        self.children[-1].before_gneb_callback   = self.before_gneb_callback
        self.children[-1].before_llg_callback    = self.before_llg_callback
        self.children[-1].n_iterations_check     = self.n_iterations_check
        self.children[-1].n_checks_save          = self.n_checks_save
        self.children[-1].allow_split            = self.allow_split

        self.child_indices.append([i1, i2])

        # Write the chain file
        chain_write_between(p_state, self.children[-1].chain_file, i1, i2)

        return self.children[-1] # Return a reference to the child that has just been added


    def spawn_children(self, p_state):
        """Creates child nodes"""

        if not self.allow_split:
            return

        if self._converged:
            self.log("Converged")
            return

        if len(self.intermediate_minima) == 0:
            return

        self.log("Spawning children - splitting chain")

        from spirit import chain

        # Instantiate the GNEB nodes
        noi = chain.get_noi(p_state)
        # Creates a list of all indices that would be start/end points of new chains
        idx_list = [0, *self.intermediate_minima, noi-1]
        # From the previous list, creates a list of pairs of start/end points
        idx_pairs = [ (idx_list[i], idx_list[i+1]) for i in range(len(idx_list)-1) ]

        # First create all the instances of GNEB nodes
        for i1,i2 in idx_pairs:
            self.add_child(p_state, i1, i2)

    def run_children(self):
        """Execute the run loop on all children"""
        # The following list determines the order in which we run the children of this node.
        # We sort the run from largest to smallest energy barrier (hence the minus).
        # We do this to explore the most 'interesting' paths first

        idx_children_run_order = list(range(len(self.children)))

        try:
        idx_children_run_order.sort(key = lambda i : -self.children[i].current_energy_path.barrier())
        except Exception as e:
            self.log("Could not sort children run order by energy barrier. Executing in order.")
            idx_children_run_order = list(range(len(self.children)))

        for i in idx_children_run_order:
            self.children[i].run()

    def chain_rebalance(self, p_state, tol=0.25):
        """Tries to rebalance the chain while keeping the number of images constant. The closer tol is to zero, the more aggressive the rebalancing."""
        import numpy as np
        from spirit import chain, transition
        from spirit.parameters import gneb
        noi = chain.get_noi(p_state)

        idx_max  = np.argmax(self.current_energy_path.total_energy)

        delta_Rx      = [ self.current_energy_path.reaction_coordinate[i+1] - self.current_energy_path.reaction_coordinate[i] for i in range(noi-1) ]
        delta_Rx_2    = [ self.current_energy_path.reaction_coordinate[i+2] - self.current_energy_path.reaction_coordinate[i] for i in range(noi-2) ]
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

        self.noi = chain.get_noi(p_state)

        if(self.noi < self.target_noi):
            self.log("Too few images ({}). Inserting additional interpolated images".format(self.noi))

        while(self.noi < self.target_noi):
            transition.homogeneous_insert_interpolated(p_state, 1)
            self.noi = chain.get_noi(p_state)

        self.log("Number of images = {}".format(self.noi))

    def check_run_condition(self):
        """Returns True if the run loop should be continued."""
        return (not self._converged) and (len(self.intermediate_minima) <= 0) and (self.total_iterations < self.max_total_iterations or self.max_total_iterations < 0)

    def run(self):
        """Run GNEB with checks after a certain number of iterations"""

        if(len(self.children) != 0): # If not a leaf node call recursively on children
            self.run_children()
            return

        try:
            self.log("Running")

            from spirit import state, simulation, io, parameters

            with state.State(self.input_file) as p_state:
                # Set the output folder for the files created by spirit
                set_output_folder(p_state, self.output_folder, self.output_tag)
                parameters.gneb.set_convergence(p_state, self.convergence)

                # The state prepare callback can be used to change the state before execution of any other commands
                # One could e.g. use the hamiltonian API to change interaction parameters instead of relying only on the input file
                if self.state_prepare_callback:
                    self.state_prepare_callback(self, p_state)

                # Before we run we must make sure that the chain.ovf file exists now
                if not os.path.exists(self.chain_file):
                    raise Exception("Chain file does not exist!")

                # Read the file, increase up to at leas target_noi and update the energy_path (for plotting etc.)
                io.chain_read(p_state, self.chain_file)
                self.increase_noi(p_state)
                self.update_energy_path(p_state)

                # Another callback at whicht the chain actually exists now, theoretically one could set image types here
                if self.before_gneb_callback:
                    self.before_gneb_callback(self, p_state)

                try:
                    n_checks = 0 
                    while(self.check_run_condition()):

                        info = simulation.start(p_state, simulation.METHOD_GNEB, simulation.SOLVER_VP_OSO, n_iterations=self.n_iterations_check)
                        self.update_energy_path(p_state)
                        self.check_for_minima() # Writes the intermediate minima list
                        n_checks += 1
                        self.total_iterations += info.total_iterations

                        # Log some information
                        self.log("Total iterations = {}".format(self.total_iterations))
                        self.log("      max.torque = {}".format(info.max_torque))
                        self.log("      ips        = {}".format(info.total_ips))
                        self.log("      Delta E    = {}".format(self.current_energy_path.barrier()))

                        self._converged = info.max_torque < self.convergence

                        # Step callback to e.g plot chains
                        if(self.gneb_step_callback):
                            self.gneb_step_callback(self, p_state)

                        # Save the chain periodically
                        if(n_checks % self.n_checks_save == 0):
                            self.save_chain(p_state)

                        # Rebalancing
                        # self.chain_rebalance(p_state)

                except KeyboardInterrupt as e:
                    self.log("Interrupt during run loop")
                    self.save_chain(p_state)
                    if(self.exit_callback):
                        self.exit_callback(self, p_state)
                    raise e

                if self.before_llg_callback:
                    self.before_llg_callback(self, p_state)

                if len(self.intermediate_minima) > 0:
                self.log("Relaxing intermediate minima")
                for idx_minimum in self.intermediate_minima:
                    simulation.start(p_state, simulation.METHOD_LLG, simulation.SOLVER_LBFGS_OSO, idx_image = idx_minimum)

                self.update_energy_path(p_state)

                if(self.exit_callback):
                    self.exit_callback(self, p_state)

                self.save_chain(p_state)

                self.spawn_children(p_state)
                # p_state gets deleted here, it does not have to persist to run the child nodes
            self.run_children()
            self.log("Finished!")

        except Exception as e:
            self.log("Exception during 'run': {}".format(str(e))) # Log the exception and re-raise
            self.log(traceback.format_exc())
            raise e

    def clamp_and_refine(self, convergence=None, max_total_iterations=None, mode="max", apply_ci=True, target_noi=5):
        """One step of clamp and refine algorithm"""

        if target_noi % 2 == 0:
            raise Exception("target_noi must be uneven!")

        if not convergence:
            convergence = self.convergence

        if not max_total_iterations:
            max_total_iterations = self.max_total_iterations

        try:
            from spirit import state
            if(len(self.children) != 0): # If not a leaf node call recursively on children
                for c in self.children:
                    c.clamp_and_refine(convergence, max_total_iterations, mode, apply_ci, target_noi)
                return

            self.update_energy_path()

            try:
                idx_max = int(mode)
            except Exception as e:
                if mode.lower() == "max":
                    idx_max = self.current_energy_path.idx_sp()
                elif mode.lower() == "cliff":
                    grad = np.abs( np.array(self.current_energy_path.total_energy)[1:] - np.array(self.current_energy_path.total_energy)[:-1] )
                    idx_max = np.argmax(grad)
                else:
                    raise Exception("Unknown mode")

            self.log("Clamp and refine! mode = {}, idx = {}".format(mode, idx_max))

            if(idx_max == 0 or idx_max == self.noi - 1):
                self.log("Cannot clamp and refine, since idx_max (= {}) is either 0 or noi-1".format(idx_max))
                return

            with state.State(self.input_file) as p_state:
                from spirit import io
                io.chain_read(p_state, self.chain_file)
                self.add_child(p_state, idx_max-1, idx_max+1)

                # Attributes, that we dont copy
                self.children[-1].allow_split = False  # Dont allow splitting for clamp and refine
                self.children[-1].target_noi           = target_noi
                self.children[-1].convergence          = convergence
                self.children[-1].max_total_iterations = max_total_iterations

            if apply_ci:
                def before_gneb_cb(gnw, p_state):
                    from spirit import parameters
                    gnw.log("Setting image type")
                        parameters.gneb.set_climbing_falling(p_state, parameters.gneb.IMAGE_CLIMBING, idx_image=int((target_noi-1)/2))
            else:
                def before_gneb_cb(gnw, p_state):
                        pass

                self.children[-1].before_gneb_callback = before_gneb_cb

            self.run_children()

        except Exception as e:
            self.log("Exception during 'clamp_and_refine': {}".format(str(e))) # Log the exception and re-raise
            self.log(traceback.format_exc())
            raise e