from anytree import Node, NodeMixin, RenderTree 
import os

import shutil
import copy

import numpy as np
import traceback

from numpy import inf

from .data import energy_path, energy_path_from_p_state
from .util import set_output_folder
from .chain_io import chain_write_between
from datetime import datetime
import json

class GNEB_Node(NodeMixin):
    """A class that represents a GNEB calculation on a single chain. Can spawn children if cutting of the chain becomes necessary."""

    def __init__(self, name, input_file, output_folder, initial_chain_file=None, gneb_workflow_log_file=None, parent=None, children=None):
        """Constructor."""
        self.chain_file : str             = ""
        self.initial_chain_file : str     = ""
        self.input_file : str             = ""
        self.gneb_workflow_log_file : str = ""
        self.current_energy_path = object()
        self.n_iterations_check  = 5000
        self.n_checks_save       = 3
        self.total_iterations    = 0
        self.target_noi  = 10
        self.noi         = -1
        self.convergence = 1e-4
        self.path_shortening_constant = -1
        self.max_total_iterations     = -1
        self.output_folder = ""
        self.output_tag    = ""
        self.allow_split = True
        self.state_prepare_callback = None
        self.gneb_step_callback     = None
        self.exit_callback          = None
        self.before_gneb_callback   = None
        self.before_llg_callback    = None
        self.intermediate_minima = []
        self.child_indices       = []
        self._ci                 = False
        self._converged          = False
        self.history = []

        # Create output folder
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
        self.output_folder = output_folder

        # The current chain is always saved here
        self.chain_file = output_folder + "/chain.ovf"

        # If an initial chain is specified we copy it to the output folder
        if(initial_chain_file):
            if not os.path.exists(initial_chain_file):
                raise Exception("Initial chain file ({}) does not exist!".format(initial_chain_file))
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

    def setup_plot_callbacks(self):
        """Sets up callbacks such that the path is plotted."""
        from spirit import simulation, chain
        from spirit.parameters import gneb
        from spirit_python_utilities.spirit_utils import plotting, data
        import matplotlib.pyplot as plt

        def mark_climbing_image(p_state, gnw, ax):
            """Helper function that marks the climbing image in a plot."""
            import numpy as np
            image_types =  np.array([gneb.get_climbing_falling(p_state, i) for i in range(chain.get_noi(p_state))])
            idx_climbing_list = np.array(range(chain.get_noi(p_state)))[image_types == gneb.IMAGE_CLIMBING]
            if(len(idx_climbing_list) == 0):
                return
            idx_climbing = idx_climbing_list[0]
            E0 = gnw.current_energy_path.total_energy[-1]
            ax.plot( gnw.current_energy_path.reaction_coordinate[idx_climbing], gnw.current_energy_path.total_energy[idx_climbing] - E0, marker="^", color="red" )

        def before_gneb_cb(gnw, p_state):
            # gneb.set_image_type_automatically(p_state)
            simulation.start(p_state, simulation.METHOD_GNEB, simulation.SOLVER_VP, n_iterations=1)
            gnw.current_energy_path = data.energy_path_from_p_state(p_state)
            plotting.plot_energy_path(gnw.current_energy_path, plt.gca())
            mark_climbing_image(p_state, gnw, plt.gca())
            plt.savefig(gnw.output_folder + "/path_{}_initial.png".format(gnw.total_iterations))
            plt.close()
        self.before_gneb_callback = before_gneb_cb

        def step_cb(gnw, p_state):
            import numpy as np
            gnw.update_energy_path(p_state)
            plotting.plot_energy_path(gnw.current_energy_path, plt.gca())
            mark_climbing_image(p_state, gnw, plt.gca())
            plt.savefig(gnw.output_folder + "/path_{}.png".format(gnw.total_iterations))
            plt.close()

            plt.plot( np.asarray(gnw.history)[:,0], np.asarray(gnw.history)[:,1] )
            plt.xlabel("Total iterations")
            plt.ylabel("Max. torque [meV]")
            plt.yscale("log")
            plt.savefig(gnw.output_folder + "/convergence.png")
            plt.close()
            
        self.gneb_step_callback = step_cb

        def exit_cb(gnw, p_state):
            plotting.plot_energy_path(gnw.current_energy_path, plt.gca())
            mark_climbing_image(p_state, gnw, plt.gca())
            plt.savefig(gnw.output_folder + "/path_{}_final.png".format(gnw.total_iterations))
            plt.close()
        self.exit_callback = exit_cb

    def change_output_folder(self, new_output_folder, log_file=None):
        if os.path.exists(new_output_folder):
            raise Exception("Cannot change to new_output_folder, that already exists!")

        if self.parent == None:
            shutil.copytree(self.output_folder, new_output_folder) # Take care of all files we might need
            log_file = new_output_folder + "/workflow_log.txt"

        self.output_folder          = new_output_folder
        self.chain_file             = self.output_folder + "/chain.ovf"
        self.gneb_workflow_log_file = log_file
        if os.path.exists(self.initial_chain_file):
            self.initial_chain_file     = self.output_folder + "/root_initial_chain.ovf"

        self.log("Changed output folder to {}".format(new_output_folder))

        for i,c in enumerate(self.children):
            child_output_folder = self.output_folder + "/{}".format(i)
            c.change_output_folder(child_output_folder, log_file)

    def collect_chain(self, output_file):
        from spirit import state, io, chain
        self.log("Collecting chain in file {}".format(output_file))

        if os.path.exists(output_file):
            os.remove(output_file)

        def helper(p_state, node):
            node.log("    collecting...")
            # Make sure the number of images matches our current simulation state
            chain.image_to_clipboard(p_state)
            noi = io.n_images_in_file(p_state, node.chain_file)
            chain.set_length(p_state, noi)
            io.chain_read(p_state, node.chain_file)
            noi = chain.get_noi(p_state)

            i = 0
            while i < noi:
                # First we check if this images is within any of the ranges covered by the children
                is_child = False
                for idx_c, (i1, i2) in enumerate(node.child_indices):
                    if i>=i1 and i<=i2:
                        is_child = True
                        idx_child = idx_c
                        break

                # If the current idx is covered by a child node, we open up another level of recursion, else we append the image
                if is_child:
                    helper(p_state, node.children[idx_child])
                    # After collecting the child we advance the current iteration idx, so that we continue one past the last child index
                    i = node.child_indices[idx_child][1] + 1
                    # We also need to read the chain file again to return to our previous state
                    chain.image_to_clipboard(p_state)
                    chain.set_length(p_state, noi)
                    io.chain_read(p_state, node.chain_file)
                else:
                    io.image_append(p_state, output_file, idx_image=i)
                    i += 1

        with state.State(self.input_file) as p_state:
            self._prepare_state(p_state)
            helper(p_state, self)

        self.log("Done collecting chain in file")

    def to_json(self):
        json_file = self.output_folder + "/node.json"
        self.log("Saving to {}".format(json_file))

        node_dict = dict(
            name                   = str(self.name),
            chain_file             = str(self.chain_file),
            initial_chain_file     = str(self.initial_chain_file),
            input_file             = str(self.input_file),
            gneb_workflow_log_file = str(self.gneb_workflow_log_file),
            n_iterations_check     = int(self.n_iterations_check),
            n_checks_save          = int(self.n_checks_save),
            total_iterations       = int(self.total_iterations),
            target_noi             = int(self.target_noi),
            convergence            = float(self.convergence),
            max_total_iterations   = int(self.max_total_iterations),
            output_folder          = str(self.output_folder),
            output_tag             = str(self.output_tag),
            child_indices          = [(int(c1), int(c2)) for c1, c2 in self.child_indices],
            allow_split            = bool(self.allow_split)
        )

        with open(json_file, "w") as f:
            f.write(json.dumps(node_dict, indent=4))

        for c in self.children:
            c.to_json()

    @staticmethod
    def from_json(json_file, parent=None, children=None):

        with open(json_file, "r") as f:
            data = json.load(f)

        name                   = data["name"]
        chain_file             = data["chain_file"]
        initial_chain_file     = data["initial_chain_file"]
        input_file             = data["input_file"]
        gneb_workflow_log_file = data["gneb_workflow_log_file"]
        n_iterations_check     = data["n_iterations_check"]
        n_checks_save          = data["n_checks_save"]
        total_iterations       = data["total_iterations"]
        target_noi             = data["target_noi"]
        convergence            = data["convergence"]
        max_total_iterations   = data["max_total_iterations"]
        output_folder          = data["output_folder"]
        output_tag             = data["output_tag"]
        child_indices          = data["child_indices"]
        allow_split            = data["allow_split"]

        result                      = GNEB_Node(name, input_file, output_folder, None, gneb_workflow_log_file, parent=parent, children=children)
        result.n_iterations_check   = n_iterations_check
        result.n_checks_save        = n_checks_save
        result.total_iterations     = total_iterations
        result.target_noi           = target_noi
        result.convergence          = convergence
        result.max_total_iterations = max_total_iterations
        result.chain_file           = chain_file
        result.initial_chain_file   = initial_chain_file
        result.output_tag           = output_tag
        result.allow_split          = allow_split
        result.child_indices        = child_indices

        result.log("Created from json file {}".format(json_file))

        for i in range(len(child_indices)):
            new_json_file = result.output_folder + "/{}/node.json".format(i)
            GNEB_Node.from_json(new_json_file, parent=result)

        return result

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
                self._prepare_state(p_state)
                self.increase_noi(p_state)
                self.update_energy_path(p_state)

    def check_for_minima(self):
        """
           Checks the chain for intermediate minima.
        """
        self.intermediate_minima = []
        e = self.current_energy_path.total_energy
        for i in range(1, len(e) - 1): # Leave out the first and the last energy
            if(e[i-1] > e[i] and e[i+1] > e[i]):
                self.intermediate_minima.append(i)

        if(len(self.intermediate_minima) > 0):
            self.log("Found intermediate minima at: {}".format(self.intermediate_minima))

    def save_chain(self, p_state):
        """Saves the chain and overwrites the chain_file"""
        from spirit import io
        self.log("Writing chain to {}".format(self.chain_file))
        io.chain_write(p_state, self.chain_file)

    def add_child(self, p_state, i1, i2):
        self.log("Adding child with indices {} and {}".format(i1, i2))
        # Attributes that change due to tree structure
        child_name          = self.name + "_{}".format(len(self.children))
        child_input_file    = self.input_file
        child_output_folder = self.output_folder + "/{}".format(len(self.children))
        self.children      += (GNEB_Node(name = child_name, input_file = child_input_file, output_folder = child_output_folder, gneb_workflow_log_file=self.gneb_workflow_log_file, parent = self), )

        self.children[-1].current_energy_path = self.current_energy_path.split(i1, i2+1)

        # Copy the other attributes
        self.children[-1].target_noi             = self.target_noi
        self.children[-1].convergence            = self.convergence
        self.children[-1].path_shortening_constant = self.path_shortening_constant
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
        """If intermediate minima are present, relaxes them and creates child nodes"""

        from spirit import chain, simulation

        if not self.allow_split:
            return

        if len(self.intermediate_minima) == 0:
            return

        if self._converged:
            return

        self.log("Spawning children - splitting chain")

        self.log("Relaxing intermediate minima")
        for idx_minimum in self.intermediate_minima:
            simulation.start(p_state, simulation.METHOD_LLG, simulation.SOLVER_LBFGS_OSO, idx_image = idx_minimum)

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

    def _prepare_state(self, p_state):
        """Prepares the state and reads in the chain."""
        from spirit import parameters, io

        # Set the output folder for the files created by spirit
        set_output_folder(p_state, self.output_folder, self.output_tag)

        # Set the gneb convergence parameter
        parameters.gneb.set_convergence(p_state, self.convergence)

        # The state prepare callback can be used to change the state before execution of any other commands
        # One could e.g. use the hamiltonian API to change interaction parameters instead of relying only on the input file
        if self.state_prepare_callback:
            self.state_prepare_callback(self, p_state)

        # Before we run we must make sure that the chain.ovf file exists now
        if not os.path.exists(self.chain_file):
            raise Exception("Chain file does not exist!")

        if self.path_shortening_constant > 0:
            parameters.gneb.set_path_shortening_constant(p_state, self.path_shortening_constant)

        # Read the file, increase up to at leas target_noi and update the energy_path (for plotting etc.)
        io.chain_read(p_state, self.chain_file)

    def check_run_condition(self):
        """Returns True if the run loop should be continued."""
        condition_minima     = len(self.intermediate_minima) <= 0
        condition_iterations = self.total_iterations < self.max_total_iterations or self.max_total_iterations < 0
        condition_convergence = not self._converged

        if self.allow_split: # Only care about minima if splitting is allowed
            return condition_minima and condition_iterations and condition_convergence
        else:
            return condition_iterations and condition_convergence

    def run(self):
        """Run GNEB with checks after a certain number of iterations"""

        if(len(self.children) != 0): # If not a leaf node call recursively on children
            self.run_children()
            return

        try:
            self.log("Running")

            from spirit import state, simulation, io

            with state.State(self.input_file) as p_state:

                self._prepare_state(p_state)
                self.increase_noi(p_state)
                self.update_energy_path(p_state)

                # Another callback at which the chain actually exists now, theoretically one could set image types here
                if self.before_gneb_callback:
                    self.before_gneb_callback(self, p_state)

                try:
                    self.log("Starting GNEB iterations")
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

                        self.history.append([self.total_iterations, info.max_torque])

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

                self.update_energy_path(p_state)

                if self._converged:
                    self.log("Converged!")
                else:
                    self.spawn_children(p_state)

                self.save_chain(p_state)

                if(self.exit_callback):
                    self.exit_callback(self, p_state)

                # p_state gets deleted here, it does not have to persist to run the child nodes
            self.run_children()
            self.log("Finished!")

        except Exception as e:
            self.log("Exception during 'run': {}".format(str(e))) # Log the exception and re-raise
            self.log(traceback.format_exc())
            raise e

    def clamp_and_refine(self, convergence=None, max_total_iterations=None, idx_max_list=None, apply_ci=True, target_noi=5):
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
                    c.clamp_and_refine(convergence, max_total_iterations, idx_max_list, apply_ci, target_noi)
                return

            self.update_energy_path()

            # To get a list of the "interesting" images we compute the second derivative of the energy wrt to Rx
            # We initialize these lists with a dummy element so the idx counting is not offset by one
            second_deriv    = [0] # second derivative
            first_deriv_fw  = [0] # first derivative forward
            first_deriv_bw  = [0] # first derivative backward

            E  = self.current_energy_path.total_energy
            Rx = self.current_energy_path.reaction_coordinate

            for idx in range(1, self.current_energy_path.noi()-1): # idx=0 and idx=noi-1 excluded!
                grad_forward = (E[idx+1] - E[idx]) / (Rx[idx+1] - Rx[idx])
                grad_backward = (E[idx] - E[idx-1]) / (Rx[idx] - Rx[idx-1])
                second = 2 * (grad_forward - grad_backward) / ( Rx[idx+1] - Rx[idx-1] )
                first_deriv_fw.append(grad_forward)
                first_deriv_bw.append(grad_backward)
                second_deriv.append(second)

            # If not specified, build the idx_max list
            if not idx_max_list:
                idx_max_list = []
                idx = 1
                while idx<self.current_energy_path.noi()-1:
                    if first_deriv_fw[idx]<0 and first_deriv_bw[idx] > 0.01 * first_deriv_fw[idx]:
                        idx_max_list.append(idx)
                        idx += 1 # If we add something to idx_max list we increment the idx so that we do not also add the point right after that
                    idx += 1

            self.log("Clamp and refine! idx_list = {}".format(idx_max_list))

            for idx_max in idx_max_list:
                if(idx_max == 0 or idx_max == self.noi - 1):
                    self.log("Cannot clamp and refine, since idx_max (= {}) is either 0 or noi-1".format(idx_max))
                    return

                with state.State(self.input_file) as p_state:
                    self._prepare_state(p_state)
                    self.add_child(p_state, idx_max-1, idx_max+1)

                    # Attributes, that we dont copy
                    self.children[-1].allow_split          = False  # Dont allow splitting for clamp and refine
                    self.children[-1].path_shortening_constant = -1 # No path shortening for clamp and refine
                    self.children[-1].target_noi           = target_noi
                    self.children[-1].convergence          = convergence
                    self.children[-1].max_total_iterations = max_total_iterations

                    if apply_ci and not self._ci:
                        def before_gneb_cb(gnw, p_state):
                            from spirit import parameters
                            self.before_gneb_callback(gnw, p_state)
                            gnw.log("Setting image type")
                            parameters.gneb.set_climbing_falling(p_state, parameters.gneb.IMAGE_CLIMBING, idx_image=int((target_noi-1)/2))
                        self.children[-1]._ci = True
                    else:
                        def before_gneb_cb(gnw, p_state):
                            self.before_gneb_callback(gnw, p_state)

                    self.children[-1].before_gneb_callback = before_gneb_cb

            self.run_children()

        except Exception as e:
            self.log("Exception during 'clamp_and_refine': {}".format(str(e))) # Log the exception and re-raise
            self.log(traceback.format_exc())
            raise e