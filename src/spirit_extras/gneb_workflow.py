from anytree import NodeMixin
import os

import shutil

import numpy as np
import traceback

from numpy import inf

from .data import energy_path, energy_path_from_p_state
from .util import set_output_folder
from .chain_io import chain_write_between
from datetime import datetime
import uuid
import json

class GNEB_Node(NodeMixin):
    """A class that represents a GNEB calculation on a single chain. Can spawn children if cutting of the chain becomes necessary."""

    def __init__(self, name, input_file, output_folder, initial_chain_file=None, gneb_workflow_log_file=None, parent=None, children=None, scratch_folder=None):
        from spirit import simulation, io

        """Constructor."""
        self.chain_file : str             = ""
        self.chain_file_post_run : str    = ""
        self.initial_chain_file : str     = ""
        self.input_file : str             = ""
        self.gneb_workflow_log_file : str = ""
        self.uuid                = uuid.uuid1()
        self.current_energy_path = object()
        self.n_iterations_check  = 5000
        self.n_checks_save       = 3
        self.total_iterations    = 0
        self.target_noi  = -1
        self.noi         = -1
        self.convergence = 1e-7
        self.path_shortening_constant = -1
        self.max_total_iterations     = -1
        self.output_folder = ""
        self.output_tag    = ""
        self.allow_split   = False
        self.max_depth     = 1
        self.moving_endpoints       = None
        self.translating_endpoints  = None
        self.delta_Rx_left          = 1.0
        self.delta_Rx_right         = 1.0
        self.state_prepare_callback = None
        self.gneb_step_callback     = None
        self.exit_callback          = None
        self.before_gneb_callback   = None
        self.before_llg_callback    = None
        self.after_llg_callback     = None
        self.intermediate_minima = []
        self.child_indices       = []
        self.image_types         = []
        self.history             = []
        self.solver_llg  = simulation.SOLVER_LBFGS_OSO
        self.solver_gneb = simulation.SOLVER_VP_OSO
        self.chain_write_fileformat = io.FILEFORMAT_OVF_BIN
        self.input_file = input_file
        self.name       = name
        self.parent     = parent
        self._log       = True # flag to disable/enable logging
        self._converged = False

        # Create output folder
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
        self.output_folder = output_folder

        # If no scratch folder is specified, we use the output folder
        # else we create a temporary folder in the scratch partition
        if scratch_folder is None:
            self.scratch_folder = self.output_folder
        else:
            self.scratch_folder = os.path.join(scratch_folder, self.name + "#" + str(self.uuid))
            os.makedirs(self.scratch_folder)

        # The current chain is always saved here
        self.chain_file = self.scratch_folder + "/chain.ovf"

        # After a run, we copy the chain to this file
        self.chain_file_post_run = self.output_folder + "/chain.ovf"

        # If an initial chain is specified we copy it to the output folder
        if(initial_chain_file):
            if not os.path.exists(initial_chain_file):
                raise Exception("Initial chain file ({}) does not exist!".format(initial_chain_file))
            self.initial_chain_file = initial_chain_file
            shutil.copyfile(initial_chain_file, self.chain_file)

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

        if(initial_chain_file):
            self.log("Initial chain file: {}".format(initial_chain_file))

    def setup_plot_callbacks(self):
        """Sets up callbacks such that the path is plotted."""
        from spirit import simulation, chain
        from spirit.parameters import gneb
        from spirit_extras import plotting, data
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
            simulation.start(p_state, simulation.METHOD_GNEB, self.solver_gneb, n_iterations=1)
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

        def after_llg_cb(gnw, p_state):
            plotting.plot_energy_path(gnw.current_energy_path, plt.gca())
            mark_climbing_image(p_state, gnw, plt.gca())
            plt.savefig(gnw.output_folder + "/path_after_llg.png".format(gnw.total_iterations))
            plt.close()
        self.after_llg_callback = after_llg_cb

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

        # Only change chain file, if it was previously in the output folder
        if self.chain_file == self.chain_file_post_run:
            self.chain_file = self.output_folder + "/chain.ovf"
        self.chain_file_post_run = self.output_folder + "/chain.ovf"

        self.gneb_workflow_log_file = log_file
        if os.path.exists(self.initial_chain_file):
            self.initial_chain_file  = self.output_folder + "/root_initial_chain.ovf"

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
            name                     = str(self.name),
            uuid                     = str(self.uuid),
            chain_file               = str(self.chain_file),
            chain_file_post_run      = str(self.chain_file_post_run),
            initial_chain_file       = str(self.initial_chain_file),
            input_file               = str(self.input_file),
            gneb_workflow_log_file   = str(self.gneb_workflow_log_file),
            n_iterations_check       = int(self.n_iterations_check),
            n_checks_save            = int(self.n_checks_save),
            total_iterations         = int(self.total_iterations),
            target_noi               = int(self.target_noi),
            convergence              = float(self.convergence),
            max_total_iterations     = int(self.max_total_iterations),
            output_folder            = str(self.output_folder),
            scratch_folder           = str(self.scratch_folder),
            output_tag               = str(self.output_tag),
            child_indices            = [(int(c1), int(c2)) for c1, c2 in self.child_indices],
            allow_split              = bool(self.allow_split),
            max_depth                = int(self.max_depth),
            path_shortening_constant = float(self.path_shortening_constant),
            moving_endpoints         = bool(self.moving_endpoints),
            delta_Rx_left            = float(self.delta_Rx_left),
            delta_Rx_right           = float(self.delta_Rx_right),
            image_types              = [ [int(t[0]), int(t[1]) ] for t in self.image_types],
            solver_llg               = int(self.solver_llg),
            solver_gneb              = int(self.solver_gneb),
            fileformat               = int(self.chain_write_fileformat)
        )

        with open(json_file, "w") as f:
            f.write(json.dumps(node_dict, indent=4))

        for c in self.children:
            c.to_json()

    def history_to_file(self, path):
        np.savetxt(path, self.history, header="iteration, max. torque" )

    @staticmethod
    def from_json(json_file, parent=None, children=None):

        with open(json_file, "r") as f:
            data = json.load(f)

        name                            = data["name"]
        input_file                      = data["input_file"]
        gneb_workflow_log_file          = data["gneb_workflow_log_file"]
        output_folder                   = data["output_folder"]

        result                          = GNEB_Node(name, input_file, output_folder, None, gneb_workflow_log_file, parent=parent, children=children)
        result.n_iterations_check       = data.get("n_iterations_check", result.n_iterations_check)
        result.n_checks_save            = data.get("n_checks_save", result.n_checks_save)
        result.total_iterations         = data.get("total_iterations", result.total_iterations)
        result.target_noi               = data.get("target_noi", result.target_noi)
        result.convergence              = data.get("convergence", result.convergence)
        result.max_total_iterations     = data.get("max_total_iterations", result.max_total_iterations)
        result.chain_file               = data.get("chain_file", result.chain_file)
        result.initial_chain_file       = data.get("initial_chain_file", result.initial_chain_file)
        result.output_tag               = data.get("output_tag", result.output_tag)
        result.allow_split              = data.get("allow_split", result.allow_split)
        result.max_depth                = data.get("max_depth", result.max_depth)
        result.child_indices            = data.get("child_indices", result.child_indices)
        result.moving_endpoints         = data.get("moving_endpoints", result.moving_endpoints)
        result.image_types              = data.get("image_types", result.image_types)
        result.delta_Rx_left            = data.get("delta_Rx_left", result.delta_Rx_left)
        result.delta_Rx_right           = data.get("delta_Rx_right", result.delta_Rx_right)
        result.path_shortening_constant = data.get("path", result.path_shortening_constant)
        result.solver_llg               = data.get("solver_llg", result.solver_llg)
        result.solver_gneb              = data.get("solver_gneb", result.solver_gneb)
        result.chain_write_fileformat   = data.get("fileformat", result.chain_write_fileformat)

        result.log("Created from json file {}".format(json_file))

        for i in range(len(result.child_indices)):
            new_json_file = result.output_folder + "/{}/node.json".format(i)
            GNEB_Node.from_json(new_json_file, parent=result)

        return result

    def log(self, message):
        """Append a message with date/time information to the log file."""
        if not self._log:  # if log flag is not set, do nothing
            return
        now = datetime.now()
        current_time = now.strftime("%m/%d/%Y, %H:%M:%S")
        log_string = "{} [{:^35}] : {}".format(current_time, self.name, message)
        with open(self.gneb_workflow_log_file, "a") as f:
            print(log_string, file=f)

    def enable_log(self):
        self._log = True

    def disable_log(self):
        self._log = False

    def update_energy_path(self, p_state=None):
        """Updates the current energy path. If p_state is given we just use that, otherwise we have to construct it first"""
        self._log = False
        if p_state:
            self.current_energy_path = energy_path_from_p_state(p_state)
            self.noi = self.current_energy_path.noi()
        else:
            from spirit import state, chain, simulation
            with state.State(self.input_file) as p_state:
                chain.update_data(p_state)
                self._prepare_state(p_state)
                simulation.start(p_state, simulation.METHOD_GNEB, self.solver_gneb, n_iterations=1) # One iteration of GNEB to get interpolated quantities
                self.update_energy_path(p_state)
        self._log = True

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
        io.chain_write(p_state, self.chain_file, fileformat=self.chain_write_fileformat)

    def backup_chain(self, path, p_state=None):
        from spirit import state, io

        """Saves the chain to a file"""
        if not p_state is None:
            self.log("Writing chain to {}".format(path))
            io.chain_write(p_state, path, fileformat=self.chain_write_fileformat)
        else:
            with state.State(self.input_file) as p_state:
                self._prepare_state(p_state)
                self.backup_chain(path, p_state)

    def add_child(self, i1, i2, p_state=None):

        def _helper(p_state):
            self.log("Adding child with indices {} and {}".format(i1, i2))
            # Attributes that change due to tree structure
            child_name          = self.name + "_{}".format(len(self.children))
            child_input_file    = self.input_file
            child_output_folder = self.output_folder + "/{}".format(len(self.children))

            if self.scratch_folder == self.output_folder:
                child_scratch_folder = None
            else:
                child_scratch_folder = self.scratch_folder

            self.children      += (GNEB_Node(name = child_name, input_file = child_input_file, output_folder = child_output_folder, scratch_folder = child_scratch_folder, gneb_workflow_log_file = self.gneb_workflow_log_file, parent = self), )
            self.children[-1].current_energy_path = self.current_energy_path.split(i1, i2+1)
            # Copy the other attributes
            self.children[-1].target_noi               = self.target_noi
            self.children[-1].convergence              = self.convergence
            self.children[-1].path_shortening_constant = self.path_shortening_constant
            self.children[-1].max_total_iterations     = self.max_total_iterations
            self.children[-1].state_prepare_callback   = self.state_prepare_callback
            self.children[-1].gneb_step_callback       = self.gneb_step_callback
            self.children[-1].exit_callback            = self.exit_callback
            self.children[-1].before_gneb_callback     = self.before_gneb_callback
            self.children[-1].before_llg_callback      = self.before_llg_callback
            self.children[-1].after_llg_callback       = self.after_llg_callback
            self.children[-1].n_iterations_check       = self.n_iterations_check
            self.children[-1].n_checks_save            = self.n_checks_save
            self.children[-1].allow_split              = self.allow_split
            self.children[-1].max_depth                = self.max_depth
            self.children[-1].chain_write_fileformat   = self.chain_write_fileformat

            self.child_indices.append([i1, i2])
            # Write the chain file
            chain_write_between(p_state, self.children[-1].chain_file, i1, i2, fileformat=self.chain_write_fileformat)

        if p_state:
            _helper(p_state)
        else:
            from spirit import state
            with state.State(self.input_file) as p_state:
                self._prepare_state(p_state)
                _helper(p_state)

        return self.children[-1] # Return a reference to the child that has just been added

    def exit(self, p_state):
        """Call this when p_state gets deleted"""
        self.save_chain(p_state)

        if(self.exit_callback):
            self.exit_callback(self, p_state)

        if self.chain_file != self.chain_file_post_run:
            self.log("Copying chain_file {} to {}".format(self.chain_file, self.chain_file_post_run))
            shutil.copyfile(self.chain_file, self.chain_file_post_run)

    def relax_intermediate_minima(self, p_state=None):
        from spirit import state, chain, simulation
        if not p_state is None:
            self.log("Relaxing intermediate minima")

            for idx_minimum in self.intermediate_minima:
                self.log(f"Relaxing {idx_minimum}")
                simulation.start(p_state, simulation.METHOD_LLG, self.solver_llg, idx_image = idx_minimum)
            chain.update_data(p_state)
            self.update_energy_path(p_state)
            if self.after_llg_callback:
                self.after_llg_callback(self, p_state)
        else:
            with state.State(self.input_file) as p_state:
                self._log = False
                self._prepare_state(p_state)
                chain.update_data(p_state)
                self.update_energy_path(p_state)
                self.check_for_minima()
                self._log = True
                self.relax_intermediate_minima(p_state)
                self.update_energy_path(p_state)
                self.exit(p_state)

    def spawn_children(self, p_state):
        """If intermediate minima are present, relaxes them and creates child nodes"""
        from spirit import chain, simulation

        if not self.allow_split or self.depth >= self.max_depth:
            return

        if len(self.intermediate_minima) == 0:
            return

        if self.before_llg_callback:
            self.before_llg_callback(self, p_state)

        self.log("Spawning children - splitting chain")
        self.relax_intermediate_minima(p_state)

        # Instantiate the GNEB nodes
        noi = chain.get_noi(p_state)
        # Creates a list of all indices that would be start/end points of new chains
        idx_list = [0, *self.intermediate_minima, noi-1]
        # From the previous list, creates a list of pairs of start/end points
        idx_pairs = [ (idx_list[i], idx_list[i+1]) for i in range(len(idx_list)-1) ]

        # First create all the instances of GNEB nodes
        for i1,i2 in idx_pairs:
            self.add_child(i1, i2, p_state)

    def run_children(self):
        """Execute the run loop on all children"""
        # The following list determines the order in which we run the children of this node.
        # We sort the run from largest to smallest energy barrier (hence the minus).
        # We do this to explore the most 'interesting' paths first

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

    def increase_noi(self, p_state=None):
        """Increases the noi by (roughly) a factor of two until the number of images is at least as large as target_noi"""
        from spirit import state, chain, transition, io

        with state.State(self.input_file) as p_state:
            self._prepare_state(p_state)
            self.noi = chain.get_noi(p_state)

            if(self.noi < self.target_noi):
                self.log("Too few images ({}). Inserting additional interpolated images".format(self.noi))

            while(self.noi < self.target_noi):
                transition.homogeneous_insert_interpolated(p_state, 1)
                self.noi = chain.get_noi(p_state)

            self.log("Number of images = {}".format(self.noi))
            self.save_chain(p_state)

    def _prepare_state(self, p_state):
        """Prepares the state and reads in the chain."""
        from spirit import parameters, io

        # Set the output folder for the files created by spirit
        set_output_folder(p_state, self.output_folder, self.output_tag)

        # Set the gneb convergence parameter
        parameters.gneb.set_convergence(p_state, self.convergence)

        if self.moving_endpoints is not None:
            parameters.gneb.set_moving_endpoints(p_state, self.moving_endpoints)
            if self.translating_endpoints is not None:
                parameters.gneb.set_translating_endpoints(p_state, self.translating_endpoints)
            if self.delta_Rx_left is not None and self.delta_Rx_right is not None:
                parameters.gneb.set_equilibrium_delta_Rx(p_state, self.delta_Rx_left, self.delta_Rx_right)

        # The state prepare callback can be used to change the state before execution of any other commands
        # One could e.g. use the hamiltonian API to change interaction parameters instead of relying only on the input file
        if self.state_prepare_callback:
            self.state_prepare_callback(self, p_state)

        # Before we run we must make sure that the chain.ovf file exists now
        if not os.path.exists(self.chain_file):
            raise Exception("Chain file does not exist!")

        if self.path_shortening_constant > 0:
            parameters.gneb.set_path_shortening_constant(p_state, self.path_shortening_constant)

        # Read the file
        io.chain_read(p_state, self.chain_file)

        # Set image types
        for idx_image, image_type in self.image_types:
            self.log("Set type of image {} to {}".format(idx_image, image_type))
            parameters.gneb.set_climbing_falling(p_state, image_type, idx_image)

    def check_run_condition(self):
        """Returns True if the run loop should be continued."""
        condition_minima     = len(self.intermediate_minima) <= 0
        condition_iterations = self.total_iterations < self.max_total_iterations or self.max_total_iterations < 0
        condition_convergence = not self._converged

        if self.allow_split and self.depth < self.max_depth: # Only care about minima if splitting is allowed
            result = condition_minima and condition_iterations and condition_convergence
        else:
            result = condition_iterations and condition_convergence

        if not result: # We are stopping and log the reason
            reason = ""
            if self.allow_split and not condition_minima:
                reason = "Intermediate minima found"
            elif not condition_iterations:
                reason = "Max. Iterations reached"
            elif not condition_convergence:
                reason = "Converged"

            self.log("Stop running. Reason: {}".format( reason ))

        return result

    def run(self):
        """Run GNEB with checks after a certain number of iterations"""
        if(len(self.children) != 0): # If not a leaf node call recursively on children
            self.run_children()
            return

        try:
            self.log("Running")

            from spirit import state, simulation, io
            self._converged = False
            self.increase_noi()

            with state.State(self.input_file) as p_state:
                self._prepare_state(p_state)
                self.update_energy_path(p_state)

                # Another callback at which the chain actually exists now, theoretically one could set image types here
                if self.before_gneb_callback:
                    self.before_gneb_callback(self, p_state)

                try:
                    self.log("Starting GNEB iterations")
                    n_checks = 0
                    while(self.check_run_condition()):

                        info = simulation.start(p_state, simulation.METHOD_GNEB, self.solver_gneb, n_iterations=self.n_iterations_check)
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
                    self.exit(p_state)
                    raise e

                self.update_energy_path(p_state)

                if not self._converged:
                    self.spawn_children(p_state)

                self.exit(p_state)
                # p_state gets deleted here, it does not have to persist to run the child nodes

            self.run_children()
            self.log("Finished!")

        except Exception as e:
            self.log("Exception during 'run': {}".format(str(e))) # Log the exception and re-raise
            self.log(traceback.format_exc())
            raise e

    def prepare_moving_endpoints(self, idx_mid = -1):
        from spirit import chain, io, state, parameters

        self.log("Preparing for moving endpoints")

        self.target_noi       = 3
        self.moving_endpoints = True
        self.image_types      = [[1, parameters.gneb.IMAGE_CLIMBING]]

        with state.State(self.input_file) as p_state:
            self._prepare_state(p_state)
            self.update_energy_path(p_state)

            noi = chain.get_noi(p_state)

            if idx_mid < 0:
                E = chain.get_energy(p_state)
                idx_mid = np.argmax(E)

            self.log("idx_mid = {}".format(idx_mid))

            if(idx_mid >= 1 and idx_mid < noi-1):
                for i in range(idx_mid-1):
                    chain.delete_image(p_state, idx_image=0)
                for i in range(noi - idx_mid - 2):
                    chain.pop_back(p_state)

            self.update_energy_path(p_state)
            self.save_chain(p_state)

    def prepare_dimer(self, idx_left, idx_right=None):
        from spirit import chain, io, state, parameters

        if idx_right is None:
            idx_right = idx_left + 1

        self.log("Preparing for dimer endpoints")

        self.target_noi            = 2
        self.moving_endpoints      = True
        self.translating_endpoints = True
        self.image_types           = []

        with state.State(self.input_file) as p_state:
            self._prepare_state(p_state)
            self.update_energy_path(p_state)

            noi = chain.get_noi(p_state)

            self.log("idx_left = {}, idx_right = {}".format(idx_left, idx_right))

            # Delete all images to the right of idx right
            for i in range(noi - idx_right - 1):
                chain.pop_back(p_state)

            # Delete all images to the left of idx_left
            for i in range(idx_left):
                chain.delete_image(p_state, idx_image=0)

            # Delete images between idx_left and idx_right
            for i in range(idx_right - idx_left - 1):
                chain.delete_image(p_state, idx_image=1)

            self.update_energy_path(p_state)
            self.save_chain(p_state)