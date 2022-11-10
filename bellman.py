# -*- coding: utf-8 -*-

import numpy as np
from numpy.random import rand
from timeit import default_timer as timer

# En este script implementamos F(x,y).

def hacer_estocastica(arg):
    return (arg.T / np.sum(arg, axis=1)).T
    
# x es la política. Es de tamano SxA
# y es la ley de transicion. Es de tamaño SxAxS
# c es una matriz de tamano SxA.
# v es un vector de tamano S.
def F(v, x, y, c, lamb):
    temp = x*(c + lamb*np.dot(y, v))
    return np.sum(temp, axis=1)

if __name__ == '__main__':
    N = 30
    S = 400
    A = 4
    
    x = rand(S,A)
    x = hacer_estocastica(x)
    y = np.array([hacer_estocastica(row) for row in rand(S,A,S)])
    c = rand(S,A)
    v = rand(S)
    
    t0 = timer()
    res = F(v, x, y, c, 0.5)
    t1 = timer()
    
    print(t1 - t0)
    
    
    