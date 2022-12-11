# -*- coding: utf-8 -*-
import numpy as np
import pickle
from numpy.random import rand
#from mdptoolbox.example import forest
#from mdptoolbox.mdp import PolicyIteration
from fom import *

def hacer_estocastica(arg):
    return (arg.T / np.sum(arg, axis=1)).T

def generar_yn(N, A, S, P):
    estados = np.arange(S)
    for s in range(S):
        yn = []
        for i in range(N):
            y_item = np.zeros((A,S))
            for a in range(A):
                probabilidades = P[a,s]
                s_next = np.random.choice(estados, p=probabilidades)
                y_item[a, s_next] = 1.0
                
            yn.append(y_item)
        
        yn = np.array(yn)
            
        with open(ruta+'yn{}.pkl'.format(s), 'wb') as f:
            pickle.dump(yn, f)

def generar_yn2(N, A, S, P, ruta, per=0.05):
    for s in range(S):
        yn = np.array([(1-per)*P[:,s,:] + per*hacer_estocastica(rand(A,S)) for _ in range(N)])
        with open(ruta+'yn{}.pkl'.format(s), 'wb') as f:
            pickle.dump(yn, f)

if __name__ == '__main__':
    ruta = 'memoria/nominal/'
    S = 50*50
    A = 4
    N = 5
    lamb = 0.6
    theta = 0.5
    k = 10
    
    P = np.array([hacer_estocastica(rand(S,S)) for _ in range(A)])
    #P, R = forest(S=S)
    
# =============================================================================
#     pi = PolicyIteration(P, R, lamb)
#     pi.run()
#     
#     pol = pi.policy
#     print("Politica con PI")
#     print(pol)
# 
#     v_final, x_final, erroresv, xf = FOM(k, S, A, N, lamb, -R, ruta_yn=rutayn, theta=theta)  
#     
#     pol_fin = leer_pol(k)
#     print("Politica con fom")
#     print(pol_fin)
# =============================================================================

    generar_yn2(N, A, S, P, ruta = ruta, per=0.01)
    
    print("Listo")



