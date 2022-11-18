import json, os, pprint, shutil


class Calculation_Folder(os.PathLike, dict):
    """Represents one folder of a calculation."""

    def __init__(self, output_folder, create=False, descriptor_file="descriptor.json"):
        self.output_folder = os.path.normpath(os.path.abspath(output_folder))

        # We enforce that the descriptor file is at the root of the output folder
        if (
            not os.path.normpath(os.path.dirname(self.to_abspath(descriptor_file)))
            == self.output_folder
        ):
            raise Exception(
                "The descriptor file has to be at the root of the calculation folder!"
            )

        self._descriptor_file_name = os.path.normpath(descriptor_file)

        self._lock_file = "~lock"

        if not os.path.isdir(self.output_folder):
            if not create:
                raise Exception(
                    f"Directory {self.output_folder} either does not exist or is a file that is not a directory. Call constructor with 'create=True' to create it."
                )
            else:
                os.makedirs(self.output_folder)

        self.from_json()

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

    def from_json(self):
        if os.path.exists(self.to_abspath(self._descriptor_file_name)):
            with open(self.to_abspath(self._descriptor_file_name), "r") as f:
                super().__init__(json.load(f))

    def to_json(self):
        descriptor_file = self.to_abspath(self._descriptor_file_name)
        file, ext = os.path.splitext(self._descriptor_file_name)

        # For safety reasons we first write to a temporary file, the main reason is that open(..., "w") will immediately truncate the descriptor file
        temporary_file = self.to_abspath(file + "__temp__" + ext)
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
