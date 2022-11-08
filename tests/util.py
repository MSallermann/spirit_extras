from ..spirit_utils import import_spirit

SPIRIT_ROOT_PATH = "~/Coding"


def choose_spirit(x):
    return (
        "5840c4564ba5275dc79aa7ae4cd4e79d6cf63b87".startswith(x.revision)
        and x.openMP
        and x.pinning
    )  # check for solver info revision and openMP and pinning


spirit_info = import_spirit.find_and_insert(
    SPIRIT_ROOT_PATH, stop_on_first_viable=True, choose=choose_spirit
)[0]
print("Imported spirit:")
print(spirit_info)
