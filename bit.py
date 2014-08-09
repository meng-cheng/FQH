#Bit manipulations
#One can define a class, but there are overheads. Memory is the problem.

def bit_get(n, index):
    return (n >> index) & 1 

def bit_set(n, index, value):
    value    = (value & 1L)<<index
    mask     = (1L) << index
    return  (n & ~mask) | value

def bit_test(n, offset):
    mask = 1 << offset
    return bool(n & mask)
    
def bit_toggle(n, offset):
    mask = 1 << offset
    return (n ^ mask)
  
def bit_getslice(n, start, end):
    mask = 2L**(end - start) -1
    return (n >> start) & mask
    
def bit_setslice(n, start, end, value):
    mask = 2L**(end - start) -1
    value = (value & mask) << start
    mask = mask << start
    n = (n & ~mask) | value
    return (n >> start) & mask

def bit_str(n):
    return str(n) if n <= 1 else bit_str(n >> 1) + str(n & 1)
