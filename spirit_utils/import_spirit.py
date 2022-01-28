class spirit_info:
    path : str = ""
    version : str = ""
    version_major : int = -1
    version_minor : int = -1
    patch : int = -1
    revision : str = ""
    openMP : bool = False
    cuda : bool = False
    pinning : bool = False
    fftw : bool = False
    defects : bool = False
    has_libspirit : bool = False
    idx: int = -1

    def __str__(self):
        return '\n'.join(('    {} = {}'.format(item, self.__dict__[item]) for item in self.__dict__))

    def insert_in_path(self, idx=0):
        import sys
        sys.path.insert(idx, self.path)


def find_spirit(base_dir="~", quiet=True, choose = lambda c : c.has_libspirit, stop_on_first_viable=True):
    import os
    from importlib.machinery import SourceFileLoader

    """ 
    Searches the directory tree starting from 'base_dir' and finds all instances of the spirit python library.
    """

    def pr(msg):
        if not quiet:
            print(msg)

    def str2bool(v):
        return v.lower() in ("yes", "true", "on", "1")

    candidate_list = []

    n_rejected_by_choose = 0
    for root, dirs, files in os.walk(os.path.expanduser(base_dir)):
        for file in files:
            if file=="spiritlib.py": # We found the spirit python library
                candidate = spirit_info()
                candidate.path = os.path.abspath( os.path.join(root, "..") )

                # Check if the shared library exists
                spiritlib = SourceFileLoader("module.name", os.path.join(root, file) ).load_module()

                if(os.path.exists(os.path.join(root, spiritlib._get_spirit_lib_name()))):
                    candidate.has_libspirit = True
                    try:
                        version = SourceFileLoader("module.name", os.path.join(root, "version.py")).load_module() # Read out the version
                        candidate.version = version.version
                        candidate.version_major = version.version_major
                        candidate.version_minor = version.version_minor
                        candidate.version_patch = version.version_patch
                        candidate.revision = version.revision
                        candidate.openMP   = str2bool(version.openmp)
                        candidate.cuda     = str2bool(version.cuda)
                        candidate.fftw     = str2bool(version.fftw)
                        candidate.pinning  = str2bool(version.pinning)
                        candidate.defects  = str2bool(version.defects)
                    except:
                        pass
                else:
                    candidate.has_libspirit = False

                if not choose(candidate):
                    n_rejected_by_choose += 1
                elif(choose(candidate) and candidate.has_libspirit):
                    candidate.idx = len(candidate_list)
                    candidate_list.append(candidate)

                break # We break here because, we dont need to evaluate any other file than "spiritlib.py"

        if len(candidate_list) > 0 and stop_on_first_viable:
            break

    if(len(candidate_list) > 1):
        pr("Found {} viable candidates".format(len(candidate_list)))
        for i,c in enumerate(candidate_list):
            pr("---")
            pr(c)
            pr("---")

    return candidate_list, n_rejected_by_choose # Or rather raise an exception?


class Candidates_Exception(Exception):
    pass


def find_and_insert(base_dir="~", quiet=False, choose=lambda c : c.has_libspirit, stop_on_first_viable=True):
    """ 
    Searches the directory tree starting from 'base_dir' and inserts the path to the spirit library into the python path.
    Raises Candidates_Exception if no viable candidates are found or if more than one viable candidate is found.
    """

    candidates, n_rejected = find_spirit(base_dir, quiet, choose, stop_on_first_viable)

    if len(candidates) > 1:
        raise Candidates_Exception("Too many viable candidates! Specify 'choose' function or set `stop_on_first_viable = True`")

    if len(candidates) == 0:
        raise Candidates_Exception("No viable candidates! ({} rejected by choose function)`".format(n_rejected))

    candidates[0].insert_in_path()

    return candidates