import json, os

class Calculation_Folder:
    """Represents one folder of a calculation."""

    def __init__(self, output_folder, create=False, descriptor_file = "descriptor.json"):
        self.output_folder            = output_folder
        self._descriptor_file_name    = descriptor_file
        self.descriptor = {}

        self._lock_file = os.path.join(self.output_folder, "~lock")

        if not os.path.isdir(self.output_folder):
            if not create:
                raise Exception("Directory {} either does not exist or is a file that is not a directory. Call with 'create=True' to create it.".format(self.output_folder))
            else:
                os.makedirs(self.output_folder)

        self.from_json()

    def __getitem__(self, key):
        return self.descriptor[key]

    def __setitem__(self, key, value):
        self.descriptor[key] = value

    def get_descriptor_file_path(self):
        return os.path.join(self.output_folder, self._descriptor_file_name)

    def from_json(self):
        if os.path.exists(self.get_descriptor_file_path()):
            with open(self.get_descriptor_file_path(), "r") as f:
                self.descriptor = json.load(f)

    def lock(self):
        """Checks for lockfile in folder. If no lock file is present the lock file is created and True is returned. Can be used to signal to other processes"""
        if not os.path.exists(self._lock_file):
            with open(self._lock_file, "w") as f:
                pass
            return True
        else:
            return False

    def locked(self):
        """Check if folder is locked."""
        if not os.path.exists(self._lock_file):
            return False
        else:
            return True

    def unlock(self):
        """Unlocks."""
        if os.path.exists(self._lock_file):
            os.remove(self._lock_file)
            return True
        else:
            return False

    def _replace_from_dict(self, string, dict):
        import re

        pattern_str = "\{[a-zA-Z0-9]+(?::[^\}^:]*)?\}"
        pattern = re.compile(pattern_str)
        m = pattern.findall(string)

        for temp in m:
            split_match = temp[1:-1].split(':')
            key_match = split_match[0]

            if len(split_match) == 2: ## if len(2) we have a format string to deal with
                format_string = split_match[1]
                literal = "{0:" + format_string + "}"
            else:
                literal = "{0}"

            key = split_match[0]
            replace = literal.format(dict[key])
            string = string.replace( temp, replace)

        return string

    def format(self, str):
        """Formats a string based on the calculation descriptor.
        Example:
            calc.descriptor = { "number": 123.23424, "name" : "bob"}
            calc.format('my_string_{key:.3f}_{name}') = 'my_string_123.234_bob'
        """
        return self._replace_from_dict(str, self.descriptor)

    def to_abspath(self, relative_path):
        return os.path.join(self.output_folder, relative_path)

    def to_relpath(self, absolute_path):
        return os.path.relpath(absolute_path, self.output_folder)

    def to_json(self):
        with open(self.get_descriptor_file_path(), "w") as f:
            f.write(json.dumps(self.descriptor, indent=4))