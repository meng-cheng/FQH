# Diagonalizing FQH in LLL on a torus. Both single layer & double layer
# Optimization using bitwise operations

import itertools
import numpy as np
import scipy as sp
import scipy.sparse.linalg as lin
import pylab
from bit import *

def config2bits(config):
    return sum([1L << n for n in config])

def bits2config(n, Ns):
# Ns is the number of orbitals
    return [bit_get(n, i) for i in range(Ns)]

def getBasis0(Ns, N):
# No momentum conservation
    states = [tuple(sorted(list(x))) for x in itertools.combinations(range(Ns), N)]
#    for ind, state in enumerate(states):
#        tab[tuple(state)] = ind 
    return states

def getBasis(Ns, N):
    sectors =  [ [] for i in range(Ns)]
    for config in itertools.combinations(range(Ns), N):
        k = sum(config)%Ns
        sectors[k].append(config2bits(config))            
    return sectors

def getBasisDL(Ns, N):
# Ns is the number of orbitals in each layer. There are 2*Ns orbitals in total
# For each orbital there is a layer index, (j, 0) or (j, 1)
# (j, s) is indexed as 2j+s
# N is the total number of electrons
    sectors = [ [] for i in range(Ns)]
    for config in itertools.combinations(range(2*Ns), N):
        k = sum([n//2 for n in config])%Ns
        sectors[k].append(config2bits(config))            
    return sectors
    
# The basis states are stored in a dict. Provide method for looking up

def fermion_sign(i, j, state):
# 0 <= i, j < Ns
    sgn = 1
    if i < j:
        for n in range(i,j):
            sgn = sgn*(2*bit_get(state, n) - 1)
    else:
        for n in range(j,i):
            sgn = sgn*(2*bit_get(state, n) - 1)
        sgn = -sgn
    return sgn

def hopping(i, j, states, btab):
# c_i^\dag c_j, i != j
    elems = []
    for ind, state in enumerate(states):
        n = state
        if bit_test(n, j) and (not bit_test(n, i)):
            bit_toggle(n, j)
            bit_toggle(n, i)
            elems.append([fermion_sign(i, j, n), btab[n], ind])
    return elems
    
def pairhopping(i1, j1, i2, j2, states, btab):
# c_{i1}^\dag c_{j1} c_{i2}^\dag c_{j2}
    elems = []
    for ind, state in enumerate(states):
        n = state
        if bit_test(n, j2) and (not bit_test(n, i2)):
            sgn = fermion_sign(i2, j2, n)
            bit_toggle(n, j2)
            bit_toggle(n, i2)
            if bit_test(n, j1) and (not bit_test(n, i1)):
                sgn *= fermion_sign(i1, j1, n)
                bit_toggle(n, j1)
                bit_toggle(n, i1)
                elems.append([sgn, btab[n], ind])
    return elems

def V(k, m, a, b, Ns):
# a*b=2*pi*Ns
    cutoff = 50
    v = 0
    for nx in range(-cutoff, cutoff+1):
        for ny in range(1,cutoff+1):
            q = np.sqrt((2*np.pi*(Ns*nx+k)/a)**2+(2*np.pi*ny/b)**2)
            v += 2*(1/q)*np.exp(-0.5*(q**2))*np.cos(2*np.pi*m*ny/Ns)
    return v/Ns

def incDL(i, n, Ns):
    return 2*((i//2+n)%Ns) + i%2

def sparse_mat_wrap(dim, elems):
    data = np.array([elem[0] for elem in elems])
    row = np.array([elem[1] for elem in elems])
    col = np.array([elem[2] for elem in elems])
    smat = sp.sparse.csr_matrix((data, (row,col)), shape=(dim,dim))
    return smat
    
    
def fqh(Ns, N, a, numE):
#Hamiltonian reads 
#\sum_{j=0}^{Ns-1} \sum_{k>|m|}c_{j+k}^\dag c_{j+m}^\dag c_{j+k+m}c_j
    sectors = getBasis(Ns, N)
    spec = []
    
    vk0 = [ V(i, 0, a, 2*np.pi*Ns/a, Ns) for i in range(Ns//2)]
    vkm = np.zeros((Ns//2, Ns//2))
    for m in range(1,Ns//2):
        for n in range(m+1, Ns//2):
            vkm[n][m] = V(n, m, a, 2*np.pi*Ns/a, Ns)
            
    for k, sector in enumerate(sectors):
        #print "basis:", sector
        dim = len(sector)
        print "COM momentum:", k
        print "Hilbert space dimension:", dim
        #create a look-up table
        tab = {}
        for ind, state in enumerate(sector):
            tab[state] = ind 
        #ham = np.zeros((dim, dim))
        #hamhop = np.zeros((dim,dim))
        ham = sp.sparse.csr_matrix((dim, dim))
        hamhop = sp.sparse.csr_matrix((dim, dim))        
        #calculate the electrostatic interaction energy, m=0
        mat = []
        for ind, bitstate in enumerate(sector):
            n = bits2config(bitstate, Ns)
            int_energy = 0.0
            for k in range(Ns//2):
                for i in range(Ns):
                    int_energy += vk0[k]*n[i]*n[(i+k)%Ns]
            mat.append((int_energy, ind, ind))
        ham = ham + sparse_mat_wrap(dim, mat)
        
        print "Finish electrostatic interactions"
        
        #calculate the m=1,2, ..., [Ns/2] hopping term
        for m in range(1,Ns//2):
            for n in range(m+1, Ns//2):
                #vkm = V(n, m, a, 2*np.pi*Ns/a, Ns)
                for i in range(2*Ns):
                    hopmat = pairhopping((i+m)%Ns, i, (i+n)%Ns, (i+n+m)%Ns, sector, tab)
                    hamhop = vkm[n][m]*sparse_mat_wrap(dim, hopmat)
                    ham = ham + hamhop + hamhop.transpose()
        
        print "Hamiltonian constructed."
        w = lin.eigsh(ham,k=numE,which="SA",maxiter=100000,return_eigenvectors=False)
        print sorted(w)
        spec.append(sorted(w))
    return spec
        
def fqhDL(Ns, N, a, t, numE):
# double layer FQHE, using sparse matrices
# Ns: number of orbitals in each layer
# N: number of electrons
# Lx: length of the torus in x direction. Notice that Lx*Ly=2*pi*Ns
# d: separation between the layers. Enter the Das Sarma-Zhang potential
# t: tunneling amplitude between the layers

    sectors = getBasisDL(Ns, N)
    spec = []
    for k, sector in enumerate(sectors):
        #print "basis:", sector
        dim = len(sector)
        print "COM momentum:", k
        print "Hilbert space dimension:", dim
        #create a look-up table
        tab = {}
        for ind, state in enumerate(sector):
            tab[tuple(state)] = ind 
        #ham = np.zeros((dim, dim))
        #hamhop = np.zeros((dim,dim))
        ham = sp.sparse.csr_matrix((dim, dim))
        hamhop = sp.sparse.csr_matrix((dim, dim))        
        #calculate the electrostatic interaction energy, m=0
        mat = []
        vk0 = [ V(i, 0, a, 2*np.pi*Ns/a, Ns) for i in range(Ns//2)]
        for ind, state in enumerate(sector):
            n1 = [0]*Ns # configuration in layer 1
            n2 = [0]*Ns # configuration in layer 2
            for p in state:
                if p%2 == 0:
                    n1[p//2] = 1
                else:
                    n2[p//2] = 1
            int_energy = 0.0
            for k in range(Ns//2):
                for i in range(Ns):
                    int_energy += vk0[k]*n1[i]*n1[(i+k)%Ns] + vk0[k]*n2[i]*n2[(i+k)%Ns]
            mat.append((int_energy, ind, ind))
        ham = ham + sparse_mat_wrap(dim, mat)
        
        print "Finish electrostatic interactions"
        
        #calculate the m=1,2, ..., [Ns/2] hopping term
        for m in range(1,Ns//2):
            for n in range(m+1, Ns//2):
                vkm = V(n, m, a, 2*np.pi*Ns/a, Ns)
                for i in range(2*Ns):
                    hopmat = pairhopping(incDL(i, m, Ns), i, incDL(i, n, Ns), incDL(i, n+m, Ns), sector, tab)
                    hamhop = vkm*sparse_mat_wrap(dim, hopmat)
                    ham = ham + hamhop + hamhop.transpose()
        
        #calculate inter-layer hopping
        for i in range(Ns):
            hopmat = hopping(2*i, 2*i+1,sector, tab)
            hamhop = -t*sparse_mat_wrap(dim, hopmat)
            ham = ham + hamhop + hamhop.transpose()
        
        print "Hamiltonian constructed."
        w = lin.eigsh(ham,k=numE,which="SA",maxiter=100000,return_eigenvectors=False)
        #print sorted(w)
        spec.append(sorted(w))
    return spec
        
        
                
if __name__ == "__main__":
    Ns = 30
    N = 10
    numE = 15
    ratio = 1 # a/b = ratio. 
    a = np.sqrt(ratio*2*np.pi*Ns)
    t = 0.0

    #spec = fqhDL(Ns, N, a, t, numE)
#    spec = fqh(Ns, N, a, numE)
#    momentum = [2*np.pi*i/Ns for i in range(Ns)]
#    levels = [[spec[j][i] for j in range(Ns)] for i in range(numE)]
#    pylab.figure()
#    for i in range(numE):
#        pylab.plot(momentum, levels[i],'ro')
    
    
