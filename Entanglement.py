# calculate entanglement entropy for FQH in a torus geometry
# Ref: A. Lauchli et. al., New J. Phys. 12, 075004 (2010)

import itertools
import numpy as np
import scipy as sp
import scipy.sparse.linalg as lin

def reduced_hilbert_space(states, start, end):
    
def renyi2(N, wf, start, end, states, tab):
# N: number of electrons
# wf: the wavefunction
# Ns: the total number of orbitals
# start, end: the subsystem in the orbital space
    for Na in range(N+1):
        for config in itertools.combinations(range(start,end), Na):
        