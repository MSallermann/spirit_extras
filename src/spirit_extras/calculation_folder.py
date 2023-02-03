import json, os, pprint, shutil, yaml, glob


class Calculation_Folder(os.PathLike, dict):
    """Represents one folder of a calculation."""

    DEFAULT_DESC_FILE = "descriptor.json"

    def __init__(self, output_folder, create=False, descriptor_file=None):
        self.output_folder = os.path.normpath(os.path.abspath(output_folder))
        self._created = False
        if not os.path.isdir(self.output_folder):
            if not create:
                raise Exception(
                    f"Directory {self.output_folder} either does not exist or is a file that is not a directory. Call constructor with 'create=True' to create it."
                )
            else:
                self._created = True
                os.makedirs(self.output_folder)

        self._lock_file = "~lock"

        self.infer_descriptor_file(descriptor_file)

        self.from_desc()

    def infer_descriptor_file(self, descriptor_file_constructor_argument):
        # We enforce that the descriptor file is at the root of the output folder
        if not descriptor_file_constructor_argument is None:
            descriptor_file_path = os.path.normpath(
                os.path.dirname(self.to_abspath(descriptor_file_constructor_argument))
            )
            if not descriptor_file_path == self.output_folder:
                raise Exception(
                    "The descriptor file has to be at the root of the calculation folder!"
                )

        # Deal with unspecified descriptor file
        if descriptor_file_constructor_argument is None:
            if self._created:  # Use default value when creating
                descriptor_file_constructor_argument = self.DEFAULT_DESC_FILE
            else:  # Try to infer descriptor file
                paths_json = glob.glob(os.path.join(self, "*.json"))
                paths_yaml = glob.glob(os.path.join(self, "*.yaml"))
                paths_yml = glob.glob(os.path.join(self, "*.yml"))

                n_files_found = len(paths_json) + len(paths_yaml) + len(paths_yml)
                if n_files_found == 0:
                    descriptor_file_constructor_argument = self.DEFAULT_DESC_FILE
                elif n_files_found == 1:
                    for p in [paths_json, paths_yaml, paths_yml]:
                        if len(p) == 1:
                            descriptor_file_constructor_argument = p[0]
                else:
                    raise Exception(
                        f"Error when trying to infer descriptor files. Multiple *.json, *.yaml or *.yml files found in {self}:\n"
                        f".json: {paths_json}\n"
                        f".yaml: {paths_yaml}\n"
                        f".yml: {paths_yml}\n"
                        "Specify `descriptor_file_name` in constructor"
                    )

        # Split the constructor argument into name and extension
        self._descriptor_file_name = os.path.normpath(
            descriptor_file_constructor_argument
        )
        self._descriptor_file_name_base, self._descriptor_file_ext = os.path.splitext(
            self._descriptor_file_name
        )

        if self._descriptor_file_ext == ".json":
            self.use_json = True
        elif (
            self._descriptor_file_ext == ".yml" or self._descriptor_file_ext == ".yaml"
        ):
            self.use_json = False

        if self._created and self._descriptor_file_ext == "":
            raise Exception(
                "Only `.yaml`, `.yml` and `.json` are supported as descriptor file endings"
            )

    def __str__(self):
        return str(self.output_folder)

    def __fspath__(self):
        return self.output_folder

    def __add__(self, other):
        return self.output_folder + other

    def to_abspath(self, relative_path):
        return os.path.join(self, relative_path)

    def to_relpath(self, absolute_path):
        return os.path.relpath(absolute_path, self)

    def copy_file(self, source, rel_dest, create_subdirs=False):
        """Copies a file to a relative path within the calculation folder"""
        dest = self.to_abspath(rel_dest)
        dirname = os.path.dirname(dest)
        if not os.path.isdir(dirname):
            if create_subdirs:
                os.makedirs(dirname)
            else:
                raise Exception(
                    "Cannot create new subdirectory in calculation folder unless called with `create_subdirs=True`"
                )
        shutil.copyfile(source, dest)

    def info_string(self):
        res = f"Folder: '{str(self)}'\n"
        res += f"Descriptor: '{self._descriptor_file_name}'\n"
        res += pprint.pformat(self)
        return res

    def from_yaml(self):
        if os.path.exists(self.to_abspath(self._descriptor_file_name)):
            with open(self.to_abspath(self._descriptor_file_name), "r") as f:
                super().__init__(yaml.safe_load(f))

    def from_desc(self):
        if self.use_json:
            self.from_json()
        else:
            self.from_yaml()

    def to_desc(self):
        if self.use_json:
            self.to_json()
        else:
            self.to_yaml()

    def to_yaml(self, output_file=None):
        if output_file is None:
            output_file = self._descriptor_file_name

        descriptor_file = self.to_abspath(output_file)
        file, ext = os.path.splitext(output_file)

        # For safety reasons we first write to a temporary file, the main reason is that open(..., "w") will immediately truncate the descriptor file
        temporary_file = self.to_abspath(file + "__temp__" + ext)

        if ext != ".yml" and ext != ".yaml":
            raise Exception("File extension has to be `.yml` or `.yaml`")

        try:
            with open(temporary_file, "w") as f:
                f.write(yaml.dump(dict(self)))
                # If json serialization has succeeded, we can remove the old json file and rename the temporary accordingly
                if os.path.exists(descriptor_file):
                    os.remove(descriptor_file)
                os.rename(temporary_file, descriptor_file)
        except Exception as e:
            print("JSON serialization has encountered an error.")
            print(f"The original file '{descriptor_file}' has not been changed.")
            # We delete the temporary file, if it exists
            if os.path.exists(temporary_file):
                os.remove(temporary_file)
            raise e

    def from_json(self):
        if os.path.exists(self.to_abspath(self._descriptor_file_name)):
            with open(self.to_abspath(self._descriptor_file_name), "r") as f:
                super().__init__(json.load(f))

    def to_json(self, output_file=None):
        descriptor_file = self.to_abspath(self._descriptor_file_name)
        if output_file is None:
            output_file = self._descriptor_file_name

        descriptor_file = self.to_abspath(output_file)
        file, ext = os.path.splitext(output_file)

        # For safety reasons we first write to a temporary file, the main reason is that open(..., "w") will immediately truncate the descriptor file
        temporary_file = self.to_abspath(file + "__temp__" + ext)

        if ext != ".json":
            raise Exception("File extension has to be `.json`")

        try:
            with open(temporary_file, "w") as f:
                f.write(json.dumps(self, indent=4))
                # If json serialization has succeeded, we can remove the old json file and rename the temporary accordingly
                if os.path.exists(descriptor_file):
                    os.remove(descriptor_file)
                os.rename(temporary_file, descriptor_file)
        except Exception as e:
            print("JSON serialization has encountered an error.")
            print(f"The original file '{descriptor_file}' has not been changed.")
            # We delete the temporary file, if it exists
            if os.path.exists(temporary_file):
                os.remove(temporary_file)
            raise e

    def lock(self):
        """Checks for lockfile in folder. If no lock file is present the lock file is created and True is returned. Can be used to signal to other processes"""
        lock_file_path = self.to_abspath(self._lock_file)
        if not os.path.exists(lock_file_path):
            with open(lock_file_path, "w") as f:
                pass
            return True
        else:
            return False

    def locked(self):
        """Check if folder is locked."""
        lock_file_path = self.to_abspath(self._lock_file)
        if not os.path.exists(lock_file_path):
            return False
        else:
            return True

    def unlock(self):
        """Unlocks."""
        lock_file_path = self.to_abspath(self._lock_file)
        if os.path.exists(lock_file_path):
            os.remove(lock_file_path)
            return True
        else:
            return False

    def _replace_from_dict(self, string, dict):
        import re

        pattern_str = "\{[a-zA-Z0-9_#.]+(?::[^\}^:]*)?\}"
        pattern = re.compile(pattern_str)
        m = pattern.findall(string)

        for temp in m:
            split_match = temp[1:-1].split(":")

            if len(split_match) == 2:  ## if len(2) we have a format string to deal with
                format_string = split_match[1]
                literal = "{0:" + format_string + "}"
            else:
                literal = "{0}"

            key = split_match[0]
            replace = literal.format(dict[key])
            string = string.replace(temp, replace)

        return string

    def format(self, str):
        """Formats a string based on the calculation descriptor.
        Example:
            calc.descriptor = {"number": 123.23, "name" : "bob"}
            calc.format('my_string_{number:.1f}_{name}') = 'my_string_123.2_bob'
        """
        return self._replace_from_dict(str, self)
