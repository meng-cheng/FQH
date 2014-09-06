# Diagonalizing FQH in LLL on a torus. Both single layer & double layer

import itertools
import numpy as np
import scipy as sp
import scipy.sparse.linalg as lin
import pylab

def getBasis0(Ns, N):
# No momentum conservation
    states = [tuple(sorted(list(x))) for x in itertools.combinations(range(Ns), N)] 
    return states

def getBasis(Ns, N):
    sectors =  [ [] for i in range(Ns)]
    for config in itertools.combinations(range(Ns), N):
        k = sum(config)%Ns
        sectors[k].append(config)            
    return sectors

def getBasisDL(Ns, N):
# Ns is the number of orbitals in each layer. There are 2*Ns orbitals in total
# For each orbital there is a layer index, (j, 0) or (j, 1)
# (j, s) is indexed as 2j+s
# N is the total number of electrons
    sectors = [ [] for i in range(Ns)]
    for config in itertools.combinations(range(2*Ns), N):
        k = sum([n//2 for n in config])%Ns
        sectors[k].append(config)            
    return sectors
    
def addp(state, pos):
    state.insert(0, pos)
    state.sort()

def removep(state, pos):
    state.remove(pos)

def fermion_sign(i, j, state):
# 0 <= i, j < Ns
    sgn = 1
    if i == j:
        return sgn

    if i < j:
        for n in state:
            if n >= i and n < j:
                sgn = -sgn
    else:
        for n in state:
            if n >= j and n < i:
                sgn = -sgn
        sgn = -sgn
    return sgn
            

def hopping(i, j, bstates, btab):
# c_i^\dag c_j
    mat = []
    for ind, state in enumerate(bstates):
        if (j in state) and ((i == j) or (not (i in state))):
            sgn = fermion_sign(i, j, state)
            nstate = list(state)
            removep(nstate, j)
            addp(nstate, i)
            mat.append([sgn, btab[tuple(nstate)], ind])
    return mat
    
def pairhopping(i1, j1, i2, j2, bstates, btab):
# c_{i1}^\dag c_{j1} c_{i2}^\dag c_{j2}
    mat = []
    for ind, state in enumerate(bstates):
        nstate = list(state)
        if (j2 in nstate) and ((i2 == j2) or  (not (i2 in nstate))):
            sgn = fermion_sign(i2, j2, state)
            removep(nstate, j2)
            addp(nstate, i2)
            if (j1 in nstate) and ((i1 == j1) or (not (i1 in nstate))):
                sgn *= fermion_sign(i1, j1, nstate)
                removep(nstate, j1)
                addp(nstate, i1)
                mat.append([sgn, btab[tuple(nstate)], ind])
    return mat
    
def Vcob(k, m, a, b, Ns):
# intra-layer Coulomb interaction
# a*b=2*pi*Ns
    cutoff = 50
    v = 0
    delta = 0.0001
    for q1 in range(-cutoff, cutoff+1):
        for nm in range(-cutoff,cutoff+1):
            q2 = m + nm*Ns
            qx = 2*np.pi*q1/a
            qy = 2*np.pi*q2/b
            q = np.sqrt(qx*qx+qy*qy+delta)
            #if (q != 0):
            v += 1/q*(1-q*q)*np.exp(-0.5*q*q)*np.cos(2*np.pi*q1*k/Ns)
    return v/Ns

def Vzd(k, m, a, b, Ns, d):
# Zhang-Das Sarma potential for inter-layer Coulomb interactions
# 1/sqrt(r^2+d^2)
# a*b=2*pi*Ns
    cutoff = 50
    deltaq = 0.0001
    v = 0
    for q1 in range(-cutoff, cutoff+1):
        for nm in range(-cutoff,cutoff+1):
            q2 = m + nm*Ns
            qx = 2*np.pi*q1/a
            qy = 2*np.pi*q2/b
            q = np.sqrt(qx*qx+qy*qy+deltaq)
            v += np.exp(-d*q)/q*(1-q*q)*np.exp(-0.5*q*q)*np.cos(2*np.pi*q1*k/Ns)
    return v/Ns
    
def Vp(k, m, a, b, Ns):
# Pseudo-potential Hamiltonian for 1/3 Laughlin state
# a*b=2*pi*Ns
# 1-q^2, (q^4-4q^2+1)/2
    cutoff = 50
    v = 0
    for q1 in range(-cutoff, cutoff+1):
        for nm in range(-cutoff,cutoff+1):
            q2 = m + nm*Ns
            qx = 2*np.pi*q1/a
            qy = 2*np.pi*q2/b
            q = np.sqrt(qx*qx+qy*qy)
            v += (1-q*q)*np.exp(-0.5*q*q)*np.cos(2*np.pi*q1*k/Ns)
    return v/Ns

def incDL(i, n, Ns):
    return 2*((i//2+n)%Ns) + i%2

def matrixwrap(dim, elems):
    mat = np.zeros((dim, dim))
    for elem in elems:
        mat[elem[1], elem[2]] = elem[0]
    return mat

def sparse_mat_wrap(dim, elems):
    data = np.array([elem[0] for elem in elems])
    row = np.array([elem[1] for elem in elems])
    col = np.array([elem[2] for elem in elems])
    smat = sp.sparse.csr_matrix((data, (row,col)), shape=(dim,dim))
    return smat
    
    
def fqh(Ns, N, a, numE):
#V(k, m) is the pseudo-potential
    sectors = getBasis(Ns, N)
    spec = []
    #compute matrix elements of the interaction
    vkm = np.zeros((Ns, Ns))
    for m in range(Ns):
        for n in range(Ns):
            vkm[n][m] = Vcob(n, m, a, 2*np.pi*Ns/a, Ns)
            
    for k, sector in enumerate(sectors):
        dim = len(sector)
        print "COM momentum:", k
        print "Hilbert space dimension:", dim
        #create a look-up table
        tab = {}
        for ind, state in enumerate(sector):
            tab[tuple(state)] = ind 
        ham = sp.sparse.csr_matrix((dim, dim))
        hamhop = sp.sparse.csr_matrix((dim, dim))        
        
        
        #calculate the electrostatic interaction energy, m=0
        #mat = []
        #for ind, state in enumerate(sector):
        #    n = [0]*Ns # configuration in layer 1
        #    for p in state:
        #        n[p] = 1
        #    int_energy = 0.0
        #    for k in range(Ns//2):
        #        for i in range(Ns):
        #            int_energy += vk0[k]*n[i]*n[(i+k)%Ns]
        #    #print "configuration:", n, int_energy
        #    mat.append((int_energy, ind, ind))
        #ham = ham + sparse_mat_wrap(dim, mat)
        
        #print "Finish electrostatic interactions"
        
        #for m in range(1,Ns//2):
        #    for n in range(m+1, Ns//2):
        #        for i in range(Ns):
        #            hopmat = pairhopping((i+m)%Ns, i, (i+n)%Ns, (i+n+m)%Ns, sector, tab)
        #            hamhop = vkm[n][m]*sparse_mat_wrap(dim, hopmat)
        #            ham = ham + hamhop + hamhop.transpose()
        
        for n1 in range(Ns):
            for n2 in range(Ns):
                for n3 in range(Ns):
                    for n4 in range(Ns):
                        if ((n1+n2)%Ns != (n3+n4)%Ns):
                            continue
                        colmat = pairhopping(n1, n3, n2, n4, sector, tab)
                        hamhop = vkm[(n1-n3)%Ns][(n1-n4)%Ns]*sparse_mat_wrap(dim, colmat)
                        ham = ham - hamhop
                        if (n2 == n3):
                            colmat = hopping(n1, n4, sector, tab)
                            hamhop = vkm[(n1-n3)%Ns][(n1-n4)%Ns]*sparse_mat_wrap(dim, colmat)
                            ham = ham + hamhop
                              
                        
        print "Hamiltonian constructed."
        w, v = lin.eigsh(ham,k=numE,which="SA",maxiter=100000)
        print sorted(w)
        spec.append(sorted(w))
    return spec

        
def fqhDLfull(Ns, N, a, t, d):
# double layer FQH, full diagonalization
# Ns: number of orbitals in each layer
# N: number of electrons
# Lx: length of the torus in x direction. Notice that Lx*Ly=2*pi*Ns
# d: separation between the layers. Enter the Das Sarma-Zhang potential
# t: tunneling amplitude between the layers

    vkm = np.zeros((Ns, Ns))
    for m in range(Ns):
        for n in range(Ns):
            vkm[n][m] = Vcob(n, m, a, 2*np.pi*Ns/a, Ns)
 
    vzdkm = np.zeros((Ns, Ns))
    for m in range(Ns):
        for n in range(Ns):
            vzdkm[n][m] = Vzd(n, m, a, 2*np.pi*Ns/a, Ns, d)   
            
    sectors = getBasisDL(Ns, N)
    for k, sector in enumerate(sectors):
        print "COM momentum:", k
        dim = len(sector)
        print "Hilbert space dimension:", dim

        #create a look-up table
        tab = {}
        for ind, state in enumerate(sector):
            tab[tuple(state)] = ind 

        ham = np.zeros((dim, dim))
        hamhop = np.zeros((dim,dim))      
                
        # intra-layer Coulomb interactions
        
        for n1 in range(Ns):
            for n2 in range(Ns):
                for n3 in range(Ns):
                    for n4 in range(Ns):
                        if ((n1+n2)%Ns != (n3+n4)%Ns):
                            continue
                        # first layer
                        colmat = pairhopping(2*n1, 2*n3, 2*n2, 2*n4, sector, tab)
                        hamhop = vkm[(n1-n3)%Ns][(n1-n4)%Ns]*matrixwrap(dim, colmat)
                        ham = ham - hamhop
                        if (n2 == n3):
                            colmat = hopping(2*n1, 2*n4, sector, tab)
                            hamhop = vkm[(n1-n3)%Ns][(n1-n4)%Ns]*matrixwrap(dim, colmat)
                            ham = ham + hamhop
                        # second layer
                        colmat = pairhopping(2*n1+1, 2*n3+1, 2*n2+1, 2*n4+1, sector, tab)
                        hamhop = vkm[(n1-n3)%Ns][(n1-n4)%Ns]*matrixwrap(dim, colmat)
                        ham = ham - hamhop
                        if (n2 == n3):
                            colmat = hopping(2*n1+1, 2*n4+1, sector, tab)
                            hamhop = vkm[(n1-n3)%Ns][(n1-n4)%Ns]*matrixwrap(dim, colmat)
                            ham = ham + hamhop
        
        # inter-layer Coulomb interactions
        # c^\dag_{n1, 1}c^\dag_{n2, 2}c_{n3, 2}c_{n4, 1}
        for n1 in range(Ns):
            for n2 in range(Ns):
                for n3 in range(Ns):
                    for n4 in range(Ns):
                        if ((n1+n2)%Ns != (n3+n4)%Ns):
                            continue
                        colmat = pairhopping(2*n1, 2*n3+1, 2*n2+1, 2*n4, sector, tab)
                        hamhop = vzdkm[(n1-n3)%Ns][(n1-n4)%Ns]*matrixwrap(dim, colmat)
                        ham = ham - hamhop
                        if (n2 == n3):
                            colmat = hopping(2*n1, 2*n4, sector, tab)
                            hamhop = vzdkm[(n1-n3)%Ns][(n1-n4)%Ns]*matrixwrap(dim, colmat)
                            ham = ham + hamhop
                            
        #calculate inter-layer hopping
        for i in range(Ns):
            hopmat = hopping(2*i, 2*i+1,sector, tab)
            hamhop = -t*matrixwrap(dim, hopmat)
            ham = ham + hamhop + hamhop.transpose()
                
        w = np.linalg.eigvalsh(ham)
        print sorted(w)
        
        
def fqhDL(Ns, N, a, t, d, numE):
# double layer FQHE, using sparse matrices
# Ns: number of orbitals in each layer
# N: number of electrons
# a: length of the torus in x direction. Notice that a*b=2*pi*Ns
# t: tunneling amplitude between the layers
# d: separation between the layers. Enter the Das Sarma-Zhang potential
# numE: number of eigenvalues

    sectors = getBasisDL(Ns, N)

    spec = []

    vkm = np.zeros((Ns, Ns))
    for m in range(Ns):
        for n in range(Ns):
            vkm[n][m] = Vcob(n, m, a, 2*np.pi*Ns/a, Ns)
 
    vzdkm = np.zeros((Ns, Ns))
    for m in range(Ns):
        for n in range(Ns):
            vzdkm[n][m] = Vzd(n, m, a, 2*np.pi*Ns/a, Ns, d)   
        
    for k, sector in enumerate(sectors):
        #print "basis:", sector
        dim = len(sector)
        print "COM momentum:", k
        print "Hilbert space dimension:", dim
        #create a look-up table
        tab = {}
        for ind, state in enumerate(sector):
            tab[tuple(state)] = ind 
        ham = sp.sparse.csr_matrix((dim, dim))
        hamhop = sp.sparse.csr_matrix((dim, dim))        
        
        orbs = []
        for n1 in range(Ns):
            for n2 in range(Ns):
                for n3 in range(Ns):
                    for n4 in range(Ns):
                        if ((n1+n2)%Ns == (n3+n4)%Ns):
                            orbs.append((n1, n2, n3, n4))
                            
        print "Done orbitals counting for two-body interactions."
        
        for orb in orbs:
            # intra-layer Coulomb interactions
            # first layer
            n1 = orb[0]
            n2 = orb[1]
            n3 = orb[2]
            n4 = orb[3]
            colmat = pairhopping(2*n1, 2*n3, 2*n2, 2*n4, sector, tab)
            hamhop = vkm[(n1-n3)%Ns][(n1-n4)%Ns]*sparse_mat_wrap(dim, colmat)
            ham = ham - hamhop
            if (n2 == n3):
                colmat = hopping(2*n1, 2*n4, sector, tab)
                hamhop = vkm[(n1-n3)%Ns][(n1-n4)%Ns]*sparse_mat_wrap(dim, colmat)
                ham = ham + hamhop
            # second layer
            colmat = pairhopping(2*n1+1, 2*n3+1, 2*n2+1, 2*n4+1, sector, tab)
            hamhop = vkm[(n1-n3)%Ns][(n1-n4)%Ns]*sparse_mat_wrap(dim, colmat)
            ham = ham - hamhop
            if (n2 == n3):
                colmat = hopping(2*n1+1, 2*n4+1, sector, tab)
                hamhop = vkm[(n1-n3)%Ns][(n1-n4)%Ns]*sparse_mat_wrap(dim, colmat)
                ham = ham + hamhop
                
            # inter-layer Coulomb interactions
            # c^\dag_{n1, 1}c^\dag_{n2, 2}c_{n3, 2}c_{n4, 1} 
            colmat = pairhopping(2*n1, 2*n3+1, 2*n2+1, 2*n4, sector, tab)
            hamhop = vzdkm[(n1-n3)%Ns][(n1-n4)%Ns]*sparse_mat_wrap(dim, colmat)
            ham = ham - hamhop
            if (n2 == n3):
                colmat = hopping(2*n1, 2*n4, sector, tab)
                hamhop = vzdkm[(n1-n3)%Ns][(n1-n4)%Ns]*sparse_mat_wrap(dim, colmat)
                ham = ham + hamhop

                        
                            
        #calculate inter-layer hopping
        for i in range(Ns):
            hopmat = hopping(2*i, 2*i+1, sector, tab)
            hamhop = -t*sparse_mat_wrap(dim, hopmat)
            ham = ham + hamhop + hamhop.transpose()
                
        print "Finish Hamiltonian construction."
        w = lin.eigsh(ham,k=numE,which="SA",maxiter=100000,return_eigenvectors=False)
        print sorted(w)
        spec.append(sorted(w))
        #break
    return spec
        
def density(v, Ns, states):
    nexpt = [0]*Ns
    for i in range(Ns):
        for ind, basis in enumerate(states):
            if i in basis:
                n = 1
            else:
                n = 0
            nexpt[i] += n*np.abs(v[ind])**2
    return nexpt
    
def plot_spec(spec):
    momentum = [2*np.pi*i/Ns for i in range(Ns)]
    levels = [[spec[j][i] for j in range(Ns)] for i in range(numE)]
    pylab.figure()
    for i in range(numE):
        pylab.plot(momentum, levels[i],'ro')     
                
if __name__ == "__main__":
    Ns = 12
    N = 6
    numE = 5
    ratio = 1 # a/b = ratio. 
    a = np.sqrt(ratio*2*np.pi*Ns)
    t = 1
    d=0

    spec0 = fqhDL(Ns, N, a, t, d, numE)
    #spec0 = fqh(Ns, N, a, numE)
    
    plot_spec(spec0)
    #spec = fqh(Ns, N, a, numE)

    
    
