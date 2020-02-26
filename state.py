###############
# state.py: State class declaration, to help with performing search or transition
#           operations on the state space.
#
# Name:     Richard (Ruochen) Wang
# UID:      504770432
###############

###############
# State class declaration
class State:
    def __init__(self, x = 2, y = 5):
        self.pos = (x, y)

    # Operator overload
    ## Returns element-wise sum of the state values
    def __add__(self, other):
        return State(self[0] + other[0], self[1] + other[1])

    # __invert__, ~ operator
    ## There is no native way to directly use a State as an index into a numpy
    ## array in python. ~A returns the state as a tuple to be used as an
    ## index into np.array() objects
    def __invert__(self):
        return (self.pos[0],self.pos[1])

    # the following two overloads lets us create set() containers for States
    # __eq__ defines the == operator
    def __eq__(self,other):
        return self.pos == other.pos

    # __hash__ defines the hash function
    def __hash__(self):
        return hash(self.pos)
    
    def __getitem__(self, index):
        return self.pos[index]
    
    def __sub__(self, other):
        return NotImplemented

    def __mul__(self, other):
        return NotImplemented

    def __int__(self):
        return [self.pos[0],self.pos[1]]
