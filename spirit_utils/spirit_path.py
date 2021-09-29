import sys, os
from importlib.machinery import SourceFileLoader

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

class No_Viable_Candidates_Exception(Exception):
    pass

def find_spirit(base_dir="~", quiet=False, choose = lambda c : c.has_libspirit, stop_on_first_viable=True):
    """ Searches the directory tree starting from 'base_dir' and finds all instances of the spirit python library.
    """

    def pr(msg):
        if not quiet:
            print(msg)

    def str2bool(v):
        return v.lower() in ("yes", "true", "on", "1")

    def insert_path(path_str):
        if(len(path_str) > 0):
            sys.path.insert(0, path_str)

    candidate_list     = []
    candidate_info = []
    candidate_list_no_lib = []

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
                        version = SourceFileLoader("module.name",  os.path.join(root, "version.py")).load_module() # Read out the version
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

                if len(candidate_list) > 0:
                    break

    if(len(candidate_list) > 1):
        pr("Found {} viable candidate".format(len(candidate_list)))
        for i,c in enumerate(candidate_list):
            pr("---")
            pr(c)
            pr("---")
        # raise Exception("Too many viable candidate_list! Specify 'choose' function or set `stop_on_first_viable = True`")
        return candidate_list # Or rather raise an exception?

    if(len(candidate_list) == 0):
        raise No_Viable_Candidates_Exception("Found no viable candidate! ({} rejected by choose function)".format(n_rejected_by_choose))

    if(len(candidate_list) == 1):
        pr("Inserting at beginning of python path")
        insert_path(candidate_list[0].path)
        return candidate_list

if __name__ == "__main__":
    candidates = find_spirit("~/Coding/spirit", 0)
    (print(c) for c in candidates)
    from spirit import state