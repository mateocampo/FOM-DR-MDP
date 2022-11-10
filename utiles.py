# -*- coding: utf-8 -*-
from numpy import arange
import numpy as np

def lista(inicio, final):
    return arange(inicio, final+1)

def smto(n):
    return n*(n+1)/2

def smfromto(a,b):
    return smto(b) - smto(a-1)

def sumto2(n):
    return n*(n+1)*(2*n+1)/6

#### Como creo que debería ser

def lista_epoca(l):
    inicio = sumto2(l-1) + 1
    final = inicio + l**2 -1
    return lista(inicio, final), smfromto(inicio, final)

def ST(k):
    return smto(sumto2(k))

def lista_epoca2(l, tam=20):
    inicio = tam*(l-1) + 1
    final = inicio + tam -1
    li = lista(inicio, final)
    return li, np.sum(li)

def ST2(k, tam=20):
    return smto(tam*k)

if __name__ == '__main__':
    k = 3
    for l in lista(1, k):
        temp = lista_epoca2(l, 20)
        print(temp)