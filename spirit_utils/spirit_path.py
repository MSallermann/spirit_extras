import sys, os
from importlib.machinery import SourceFileLoader

def find_spirit(base_dir="~", choose=None, quiet=False):
    """ Searches the directory tree starting from 'base_dir' and finds all instances of the spirit python library.
        If only a single instance is found it is added to the python path. (So that from spirit impor XXX works.)
        If multiple instances are found and choose is 'None' the paths are printed and nothing happens.
        If multiple instance are found and choose has a value the choose'th instance is added to the python path.
    """
    def pr(msg):
        if not quiet:
            print(" " + msg)

    def insert_path(path_str):
        if(len(path_str) > 0):
            sys.path.insert(0, path_str)

    candidates     = []
    candidate_info = []
    candidates_no_lib = []

    for root, dirs, files in os.walk(os.path.expanduser(base_dir)):
        for file in files:
            if file=="spiritlib.py": # We found the spirit python library
                # Check if the shared library exists
                spiritlib = SourceFileLoader("module.name",  os.path.join(root,file) ).load_module()
                if(os.path.exists(os.path.join(root, spiritlib._get_spirit_lib_name()))):
                    # Read out some version information
                    candidates.append( os.path.abspath( os.path.join(root, "..") ) )
                    info = {"version" : None, "openMP" : None, "cuda" : None}
                    try:
                        version = SourceFileLoader("module.name",  os.path.join(root, "version.py")).load_module() # Read out the version
                        info = {"version" : version.version, "openMP" : version.openmp, "cuda" : version.cuda}
                    except:
                        pass
                    candidate_info.append( info )
                else:
                    candidates_no_lib.append( os.path.abspath( os.path.join(root, "..") ) )

    if(len(candidates) > 1):
        pr("Found {} candidate(s)".format(  len(candidates)))
        for i, (cand, info) in enumerate(zip(candidates, candidate_info)):
            pr( "   [{}]: {:<60} (version = {}, openMP = {}, cuda = {})".format(i, cand, info["version"], info["openMP"], info["cuda"] ))
    if(len(candidates_no_lib) > 0 and choose is None):
        pr("Additionally found {} candidate(s), without the shared library".format( len( candidates_no_lib ) ))
        for i, cand in enumerate(candidates_no_lib):
            pr( "   [{}]: {:<60}".format(i,cand) )

    spirit_path = ""
    if len(candidates) == 0:
        pr("Found no viable candidates.")
    elif len(candidates) == 1:
       spirit_path = candidates[0]
    elif len(candidates) > 1:
        if choose is None:
            pr("Use argument 'choose' to select the desired version.")
        elif choose>=0 and choose<len(candidates):
            spirit_path = candidates[choose]
        else:
            print("Invalid argument for choose")

    insert_path(spirit_path)
    print("Spirit path: {:<60}".format(spirit_path))
    return spirit_path

if __name__ == "__main__":
    find_spirit("~/Coding/spirit", 0)
    from spirit import state