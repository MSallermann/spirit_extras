import os, datetime
from termcolor import cprint


class Dependency_Gatherer:
    class Dependency_Gatherer_Exception(Exception):
        pass

    def __init__(self, verbose=True):
        self.verbose = verbose
        self._dependencies = []

        if os.name.lower() == "windows":
            os.system("color")

    def _format_time_delta(self, delta_time):
        total_seconds = abs(delta_time.total_seconds())
        weeks = int(total_seconds / (60**2 * 24 * 7))

        total_seconds -= weeks * 60**2 * 24 * 7
        days = int(total_seconds / (60**2 * 24))

        total_seconds -= days * 60**2 * 24
        hours = int((total_seconds / (60**2)))

        total_seconds -= hours * 60**2
        minutes = int((total_seconds) / 60)

        total_seconds -= minutes * 60
        seconds = int(total_seconds)

        res_string = "~"

        if weeks > 0:
            res_string += f"{weeks} week(s) and {days} day(s) ago"
        elif days > 0:
            res_string += f"{days} day(s) and {hours} hour(s) ago"
        elif hours > 0:
            res_string += f"{hours} hour(s) and {minutes} minute(s) ago"
        elif minutes > 0:
            res_string += f"{minutes} minute(s) and {seconds} second(s) ago"
        elif seconds > 3:
            res_string += f"{seconds} seconds ago"
        else:
            res_string += f"just now"

        return res_string

    def print(self, msg, color=None, text_highlights=None, indent=0):
        if color is None:
            print(indent * " " + msg)
        else:
            cprint(indent * " " + msg, color, text_highlights)

    def print_err(self, msg, indent=8):
        self.print(msg, "red", indent=indent)

    def print_success(self, msg, indent=8):
        if self.verbose:
            self.print(msg, "green", indent=indent)

    def print_warning(self, msg, indent=8):
        self.print(msg, "yellow", indent=indent)

    def print_info(self, msg, indent=8):
        if self.verbose:
            self.print(msg, indent=indent)

    def __check_path(self, path):
        """Check if path"""
        try:
            os.path.exists(path)
        except TypeError:
            return False
        return True

    def depends(self, abs_path_to_file, cb_function=None, always_generate=False):
        """Adds a dependency. If the file is not found it can optionally be created by a callback function.
        Args:
            abs_path_to_file (list of paths (list of str) or single path): Paths to the dependencies.
            cb_function (_type_, optional): Callback function that creates the files. Defaults to None.

        Returns:
            Dependcy_Gatherer: self
        """
        if self.__check_path(abs_path_to_file):
            abs_path_to_file = [abs_path_to_file]

        self._dependencies.append([abs_path_to_file, cb_function, always_generate])
        return self

    def generate(self, abs_path_to_file, cb_function):
        """Adds a dependency. The file will always be created by the callback function.

        Args:
            abs_path_to_file (_type_): Paths to the dependencies.
            cb_function (_type_): Paths to the dependencies.

        Returns:
            Dependcy_Gatherer: self
        """
        return self.depends(abs_path_to_file, cb_function)

    def check(self):
        self.print_info("Running checks...", indent=0)
        time_now = datetime.datetime.now()

        for idx, (paths, cb, always_generate) in enumerate(self._dependencies):
            # Perform existence checks
            all_exist = True
            if not always_generate:
                for idx_p, p in enumerate(paths):
                    if len(paths) == 1:
                        self.print_info(f"[{idx:^5}] Checking for '{p}'...", indent=0)
                    else:
                        if idx_p == 0:
                            self.print_info(
                                f"[{idx:^5}] Checking for {len(paths)} files...",
                                indent=0,
                            )
                        self.print_info(f"({idx_p:^5}) Checking for '{p}'...", indent=8)

                    if os.path.exists(p):
                        time_modified_epoch = os.path.getmtime(p)
                        time_modified = datetime.datetime.fromtimestamp(
                            time_modified_epoch
                        )
                        time_delta_modified = time_now - time_modified
                        time_delta_modified_string = self._format_time_delta(
                            time_delta_modified
                        )

                        self.print_success("FOUND!", indent=8)
                        self.print_info(
                            "Modified {} ({})".format(
                                time_modified.strftime("%a %b %d %H:%M:%S"),
                                time_delta_modified_string,
                            ),
                            indent=8,
                        )
                    else:
                        all_exist = False
                        self.print_warning("NOT FOUND!", indent=8)
                        if cb is None:
                            self.print_err(
                                "No callback to create file -> aborting", indent=8
                            )
                            raise self.Dependency_Gatherer_Exception(
                                f"Cannot create missing dependency: `{p}`"
                            )

            # Create files
            if always_generate or not all_exist:
                self.print_info(
                    "Trying to create files via callback function...", indent=8
                )
                try:
                    cb()
                except Exception as e:
                    self.print_err(
                        "Failure! Callback raised an exception -> aborting", indent=8
                    )
                    raise e

                # Check if files have been created successfully
                for p in paths:
                    if os.path.exists(p):  # File has to exist
                        # Modification date of file should be now
                        time_m_epoch = os.path.getmtime(p)
                        time_m = datetime.datetime.fromtimestamp(time_m_epoch)
                        time_delta = time_m - time_now
                        if time_delta.total_seconds() < -0.1:
                            self.print_err(
                                f"File `{p}` exists, but was not produced by callback (see creation date) -> aborting",
                                indent=8,
                            )
                            self.print_err(
                                f"Time_created - Time_now = {time_delta.total_seconds()} s",
                                indent=8,
                            )
                            raise self.Dependency_Gatherer_Exception(
                                f"File `{p}` exists, but was not produced by callback (see creation date)."
                            )
                    else:
                        self.print_err(
                            f"Failure! Callback did not produce file `{p}` -> aborting",
                            indent=8,
                        )
                        raise self.Dependency_Gatherer_Exception(
                            f"Callback did not produce file `{p}`"
                        )
                    self.print_success("CREATED!", indent=8)

        self.print(
            f"ALL CHECKS COMPLETE, {len(self._dependencies)} TOTAL!", "grey", "on_green"
        )
