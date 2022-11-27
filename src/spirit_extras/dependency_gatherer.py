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
        if self.verbose:
            if color is None:
                print(indent * " " + msg)
            else:
                cprint(indent * " " + msg, color, text_highlights)

    def print_err(self, msg, indent=8):
        self.print(msg, "red", indent=indent)

    def print_success(self, msg, indent=8):
        self.print(msg, "green", indent=indent)

    def print_warning(self, msg, indent=8):
        self.print(msg, "yellow", indent=indent)

    def print_info(self, msg, indent=8):
        self.print(msg, indent=indent)

    def depends(self, abs_path_to_file, cb_function=None):
        self._dependencies.append([abs_path_to_file, cb_function, False])
        return self

    def generate(self, abs_path_to_file, cb_function):
        self._dependencies.append([abs_path_to_file, cb_function, True])
        return self

    def check(self):
        self.print_info("Running checks...", indent=0)
        time_now = datetime.datetime.now()

        for idx, (path, cb, always_generate) in enumerate(self._dependencies):
            if not always_generate:
                self.print_info(f"[{idx:^5}] Checking for '{path}'...", indent=0)
            else:
                self.print_info(f"[{idx:^5}] Creating `{path}`", indent=0)

            if os.path.exists(path) and not always_generate:
                time_modified_epoch = os.path.getmtime(path)
                time_modified = datetime.datetime.fromtimestamp(time_modified_epoch)
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
                if not always_generate:
                    self.print_warning("NOT FOUND!", indent=8)
                    if cb is None:
                        self.print_err(
                            " ... No callback to create file -> aborting", indent=8
                        )
                        raise self.Dependency_Gatherer_Exception(
                            f"Cannot create missing dependency: `{path}`"
                        )
                    self.print_info("Trying to create via callback function", indent=8)

                try:
                    cb()
                except Exception as e:
                    self.print_err(
                        "Failure! Callback raised an exception -> aborting", indent=8
                    )
                    raise e

                if os.path.exists(path):  # File has to exist
                    # Modification date of file should be now
                    time_m_epoch = os.path.getmtime(path)
                    time_m = datetime.datetime.fromtimestamp(time_m_epoch)
                    time_delta = time_m - time_now
                    if time_delta.total_seconds() < -0.1:
                        self.print_err(
                            f"File `{path}` exists, but was not produced by callback (see creation date) -> aborting",
                            indent=8,
                        )
                        self.print_err(
                            f"Time_created - Time_now = {time_delta.total_seconds()} s",
                            indent=8,
                        )
                        raise self.Dependency_Gatherer_Exception(
                            f"File `{path}` exists, but was not produced by callback (see creation date)."
                        )
                else:
                    self.print_err(
                        "Failure! Callback did not produce file -> aborting", indent=8
                    )
                    raise self.Dependency_Gatherer_Exception(
                        f"Callback did not produce file `{path}`"
                    )
                self.print_success("CREATED!", indent=8)
        self.print(
            f"ALL CHECKS COMPLETE, {len(self._dependencies)} TOTAL!", "grey", "on_green"
        )
