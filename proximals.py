import numpy as np
import cvxpy as cp
from timeit import default_timer as timer
from numpy.random import rand, seed
import pickle
from scipy.optimize import bisect
import matplotlib.pyplot as plt
from bellman import hacer_estocastica

def norm2sq(x):
    return np.sum(np.square(x))

# Función para proyectar un solo vector en el simplex
def vec2simplex(vecX, l=1.):
    m = vecX.size
    vecS = np.sort(vecX)[::-1]
    vecC = np.cumsum(vecS) - l
    vecH = vecS - vecC / (np.arange(m) + 1)
    r = np.max(np.where(vecH>0)[0])
    t = vecC[r] / (r + 1)

    return np.maximum(0, vecX - t)

# Función eficiente para implementar muchos vectores en el simplex.
def mat2simplex(matX, l=1.):
    matX = matX.T
    m, n = matX.shape
    matS = np.sort(matX, axis=0)[::-1]
    matC = np.cumsum(matS, axis=0) - l
    matH = matS - matC / (np.arange(m) + 1).reshape(m,1)
    matH[matH<=0] = np.inf
    r = np.argmin(matH, axis=0)
    t = matC[r,np.arange(n)] / (r + 1)
    matY = matX - t
    matY[matY<0] = 0

    return matY.T

# La siguiente funcion sirve para verificar que prox_x funciona correctamente
# =============================================================================
# def prox_x2(g, x):
#     A = len(x)
#     xp = cp.Variable(A)
#     objetivo = cp.Minimize( g @ xp  + 0.5 * cp.sum_squares(xp - x))
#     restricciones = [xp >= 0, cp.sum(xp) == 1]
#     prob = cp.Problem(objetivo, restricciones)
#     prob.solve()
#     
#     return xp.value
# =============================================================================

# Esta es la implementación eficiente de prox_x2
def prox_x(g, x):
    return vec2simplex(x-g)

# yp es y'. De tamano NxAxS.
# yn es la y nominal. Es de tamano NxAxS para cada elemento de S
# h es de tamano AxS.
# gamma es el multiplicador de Lagrange. Luego hay que hacer binary search con el.
def prox_y_s_ant(yp, yn, h, sigma, gamma):
    N, A, S = yp.shape
    param = (sigma/(1+sigma*2*gamma)) 
    h = np.array([h for _ in range(N)])
    # Elemento a proyectar
    apy = param * ((1/sigma) * yp + 2*gamma*yn - h)
    y = mat2simplex(apy.reshape((-1,S)))
    
    return y.reshape((N,A,S))

# Esta es la mejor version que tenemos
def proxy_s(yp, yn, h, sigma, theta):
    N, A, S = yp.shape
    def f(gamma):
        y = prox_y_s_ant(yp, yn, h, sigma, gamma)
        dif_res = np.mean([norm2sq(y_res - y_nom) for y_res, y_nom in zip(y,yn)]) - theta**2
        return dif_res, y
    
    error_cero, y_cero = f(0)
    if error_cero < 0:
        return y_cero
        #raise Exception("Puede poner el valor de theta mas pequeño")
    else:
        gamma_opt = bisect(lambda x: f(x)[0], 0, 3, xtol=1e-2)
        return prox_y_s_ant(yp, yn, h, sigma, gamma_opt)

# =============================================================================
# # Esta es otra versión. Creo que es incorrecta, pero nos puede funcionar
# def proxys(yp, yn, h, sigma, N, theta):
#     def f(gamma):
#         param = (sigma/ (1+sigma*gamma)) 
#         # Elemento a proyectar
#         apy = param * ((1/sigma) * yp[0] + gamma*yn[0] - h)
#         y_cero = mat2simplex(apy)
#         dif_res = norm2sq(y_cero - yn[0]) - theta**2
#         
#         return dif_res, y_cero
#     
#     error_cero, y_cero = f(0)
#     if error_cero < 0:
#         return prox_y_s_ant(yp, yn, h, sigma, 0, N)
#     else:
#         if f(10)[0] < 0:
#             gamma_opt = bisect(lambda x: f(x)[0], 0, 10, xtol=1e-2)
#             return prox_y_s_ant(yp, yn, h, sigma, gamma_opt, N)
#         else:
#             raise Exception("No factibilidad. Elija un valor para theta más grande.")
# =============================================================================

# Hice esta funcion para entender con que se hacia bisection.
# =============================================================================
# def prox_graph(yp, yn, h, sigma, theta):
#     def f(gamma):
#         y = prox_y_s_ant(yp, yn, h, sigma, gamma)
#         dif_res = np.mean([norm2sq(y_res - y_nom) for y_res, y_nom in zip(y,yn)]) - theta**2
#         
#         dif2 = np.sum([norm2sq(y_res - y_nom) for y_res, y_nom in zip(y,yp)])
#         dif1 = np.sum([np.sum(yi*h) for yi in y])
#         
#         obj = dif1 + (1/(2*sigma))*dif2 + (gamma)*dif_res*N
#         return dif_res, obj
#     
#     xx = np.linspace(0, 3)
#     yyl = list(map(f, xx))
#     
#     yy1, yy2 = zip(*yyl)
#     
#     plt.plot(xx, yy1, label='h')
#     plt.plot(xx, 0.05*(np.array(yy2)-22), label='p')
#     plt.axvline(x=1.307, linestyle='dotted')
#     plt.xlabel('gamma')
#     plt.legend()
#     plt.grid()
#     plt.show()
# =============================================================================
    
#%%

# Ahora probamos los métodos y medimos el tiempo en el que se ejecutan
if __name__ == '__main__':
    seed(0)
    N = 30
    S = 40
    A = 5
    g = rand(A)
    yp = np.array([hacer_estocastica(row) for row in rand(N,A,S)])
    yn = np.array([hacer_estocastica(row) for row in rand(N,A,S)])
    h = rand(A,S)
    gamma = 0.3
    sigma = 0.9
    theta = 0.5

    x_inicial = rand(A)
    x_inicial = x_inicial/np.sum(x_inicial)
    
# =============================================================================
#     res1 = prox_x2(g,x_inicial)
#     res2 = prox_x(g,x_inicial)
# =============================================================================
    
    #prox_graph(yp, yn, h, sigma, theta)
    
# =============================================================================
#     ts = timer()
#     for _ in range(1000):
#         prox_x2(g, x_inicial)
#     tf = timer()
#     tiempo1 = tf - ts
#     print(tiempo1)
#     
#     ts = timer()
#     for _ in range(1000):
#         prox_x(g, x_inicial)
#     tf = timer()
#     tiempo2 = tf - ts
#     print(tiempo2)
#     
#     print(tiempo1/tiempo2)
# =============================================================================
    
# =============================================================================
#     t0 = timer()
#     for _ in range(10000):
#         resy = prox_y_s_ant(yp, yn, h, sigma, gamma)
#     t1 = timer()
#     
#     print(t1 - t0)
# =============================================================================
    
# =============================================================================
#     theta = 0.5
#     t0 = timer()
#     for _ in range(1000):
#         resy = proxy_s(yp, yn, h, sigma, theta)
#     t1 = timer()
#     
#     print(t1 - t0)
# =============================================================================
    
