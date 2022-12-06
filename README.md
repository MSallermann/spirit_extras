# spirit-extras
Package with some useful scripts, to be used together with the Spirit python API (`github.com/spirit-code/spirit/`).


## Data structure for spin systems

### Parse data from a Spirit state pointer
```python
from spirit import state
import spirit_extras.data
with state.State("input.cfg") as p_state:
    spin_sytem = spirit_extras.data.spin_system_from_p_state()

    # Information about the bravais lattice is automatically parsed from the spirit API
    print(spin_system)

    # The spins can be set
    ib, a, b, c = 1, 5, 3, 4
    idx = spin_system.idx(ib, a, b, c) # computes the linear index from a bravais tupel
    spin_system.spins[idx] = [0,0,-1]

    ib, a, b, c = spin_system.tupel(5) # computes the bravais tupel from a linear index

    # A shaped view into the spin system is also available
    spin_system.shaped().spins[ib,a,b,c] = [0,0,1]

    # Slicing syntax works
    spin_system.shaped().spins[ib,:,b,c] = [0,1,0]

    mask = np.linalg.norm(spin_system.positions - spin_system.center() < 5, radius=1)
    spin_sytem.spins[mask] = [1,0,0]
```

### Create from your own buffers
```python
# Since spirit_extras.Spin_System is needed for many functions in spirit_extras, it can also be created from a spins and positions arrays
spins = [ [1,0,0] for i in range(10) ]

# The default behaviour is to expect an unordered system, meaning some functions will throw an exception
my_spin_system = spirit_extras.data.Spin_System(positions=None, spins=spins) # positions dont _have_ to be specified
try:
    my_spin_sytem.shaped() # needs an ordered system
except:
    print("Throws :(")

# To create an ordered system pass `unordered=False` and specify `n_cells`
my_spin_system = spirit_extras.data.Spin_System(positions=None, spins=spins, n_cells=[10,1,1], unordered=False)
my_spin_sytem.shaped() # works!
```