# -*- coding: utf-8 -*-

# En este script hacemos la implementacion del Algoritmo 1 para WD - MDP

import numpy as np
from numpy.random import rand
from numpy.linalg import norm
from proximals import prox_x, proxy_s
from bellman import F, hacer_estocastica
from math import sqrt
from utiles import lista, lista_epoca, ST, lista_epoca2, ST2
from timeit import default_timer as timer
import pickle
#import concurrent.futures 
from multiprocessing import Pool
import matplotlib.pyplot as plt

rutay = 'memoria/valoresy/'
rutayn = 'memoria/nominal/'

def leer_pol(epoca):
    with open('memoria/' + 'pol_epoch{}.pkl'.format(epoca), 'rb') as f:
        x = pickle.load(f)
    x = hacer_estocastica(np.array(x))
    return np.argmax(x, axis=1)

#%%

# yn es la y nominal. Es de tamano NxAxS para cada elemento de S. Debo modificar esto.
def auxiliar(args):
    s, xs, v, N, A, lamb, c, ruta_yn, listal, ST_val, tao, sigma, theta, pesos, x_outs = args
    
    xsl = []
    ysl = []
    
    x = xs
    
    with open(ruta_yn + 'yn{}.pkl'.format(s), 'rb') as f:
        yns = pickle.load(f)
        
    with open(rutay + 'y{}.pkl'.format(s), 'rb') as f:
        y = pickle.load(f)
    
    for t in listal:
            
        cp = c[s,:] + lamb * np.mean(np.dot(y, v), axis=0)
        x_n = prox_x(tao*cp, x)
        xsl.append(x_n)
        
        x_outs += (t/ST_val) * x_n
        
        # Ahora con los y's
        h = (-lamb/N) * np.outer(2*x_n - x, v)
        #print(h.shape)
        y_n = proxy_s(y, yns, h, sigma, theta)
        ysl.append(y_n)
        
        #y_out += (t/ ST_val) * y_n
        x = x_n
        y = y_n
        
    xs_prom = np.average(xsl, axis=0, weights = pesos)
    ys_prom = np.average(ysl, axis=0, weights = pesos)
    
    return xs_prom, ys_prom, y, x_outs

def FOM(k, S, A, N, lamb, c, ruta_yn, theta):
    start = timer()
    v = rand(S)
    v_ant = v
    erroresv = []
    x = hacer_estocastica(rand(S,A))
    
    for s in range(S):
        with open(rutay + 'y{}.pkl'.format(s), 'wb') as f:
            pickle.dump(np.array([hacer_estocastica(row) for row in rand(N,A,S)]), f)
            
    x_out = np.zeros((S,A))
    #y_out = np.zeros((S,N,A,S))
    
    ST_val = ST(k)
    
    for l in lista(1,k):
        listal, Sl = lista_epoca(l)
        
        tao = pow(sqrt(A)*lamb*norm(v), -1)
        sigma = N*sqrt(A)*pow(lamb*norm(v), -1)
        
        pesos = listal / Sl
        
        def dar_argumentos_s():
            for s in range(S):
                yield s, x[s], v, N, A, lamb, c, ruta_yn, listal, ST_val, tao, sigma, theta, pesos, x_out[s]
        
        with Pool(7) as p:
            proms = p.map(auxiliar, dar_argumentos_s())
    
        x_prom, y_prom, y_vals, x_out_vals = zip(*proms)
        
        for s in range(S):
            with open(rutay + 'y{}.pkl'.format(s), 'wb') as f:
                pickle.dump(y_vals[s], f)
            
        #x_prom = np.array(x_prom)
        y_prom = np.array(y_prom)
        y_bar = np.mean(y_prom, axis=1)
        
        x_out = x_out_vals
        #print(np.sum(x_out, axis=1))
        
        x = x_prom
        v = F(v, x, y_bar, c, lamb)
        
        erroresv.append(norm(v-v_ant))
        v_ant = v
        
        with open('memoria/' + 'pol_epoch{}.pkl'.format(l), 'wb') as f:
            pickle.dump(x_out, f)
        
        print("Iteration {} ready".format(l))
        
    end = timer()
    tiempo = end - start
    print('Total time FOM', tiempo/60, 'minutos')
    
    return v, np.array(x_out), np.array(erroresv)
        
if __name__ == '__main__':
    k = 10
    theta = 0.5
    
    S = 50*50
    A = 4
    N = 5
    c = rand(S,A)
    lamb = 0.6
    
    v_final, x_final, erroresv = FOM(k, S, A, N, lamb, c, ruta_yn=rutayn, theta=theta)
    



