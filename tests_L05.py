import numpy as np
from Labo03 import norma

### Funciones L05-QR
def QR_con_GS(A,tol=1e-12,retorna_nops=False):
    """
    A una matriz de n x n 
    tol la tolerancia con la que se filtran elementos nulos en R
    retorna_nops permite (opcionalmente) retornar el numero de operaciones realizado
    retorna matrices Q y R calculadas con Gram Schmidt (y como tercer argumento opcional, el numero de operaciones).
    Si la matriz A no es de n x n, debe retornar None
    """
    n, m = A.shape
    if (n != m):
        return None
    matriz_trans = np.zeros((n,m))
    matriz_traigSup = np.zeros((n,m))
    normales = np.zeros(n)
    cant_op = 0

    for columna in range(n):
        vector_a_normalizar = np.array(n)
        for fila in range(n):
            vector_a_normalizar[fila] = A[fila][columna]
        normales[columna] = norma(vector_a_normalizar,2)

    for i in range(n):
        matriz_trans[0][i] = A[0][i]/ normales[0]
    matriz_traigSup[0][0] = normales[0]
    cant_op = 2

    for columna in range(1,m):
        for columna2 in range(columna-1):
            for fila in range(columna-1):
                matriz_traigSup[columna2][columna] = (matriz_trans[fila][columna2]*A[fila][columna])/normales[columna] 
                cant_op += 1
            for fila in range(columna-1):
                matriz_trans = A[fila][columna] - matriz_traigSup[columna2][columna]*matriz_trans[fila][columna2]
                cant_op += 1
        matriz_traigSup[columna][columna] = normales[columna]
        cant_op += 3

    return matriz_trans, matriz_traigSup, cant_op

def QR_con_HH(A,tol=1e-12,extras=False):
    """
    A una matriz de m x n (m>=n)
    tol la tolerancia con la que se filtran elementos nulos en R
    retorna matrices Q y R calculadas con reflexiones de Householder
    Si la matriz A no cumple m>=n, debe retornar None
    extras : bool, opcional
        Si es True, devuelve informacion extra sobre el proceso de factorizacion.
        Por defecto es False. Esto lo hacemos para poder graficar el proceso.
    Devuelve la factorizacion QR de A usando reflectores de Householder.
    Devuelve: 
        Q, R, extra_info (si extras es True)
        Q, R (si extras es False)
    extra_info es un diccionario con la clave:
        'R_matrices': lista de las matrices R en cada paso
        'Q_matrices': lista de las matrices Q en cada paso


    """
def calculaQR(A,metodo='RH',tol=1e-12):
    """
    A una matriz de n x n 
    tol la tolerancia con la que se filtran elementos nulos en R    
    metodo = ['RH','GS'] usa reflectores de Householder (RH) o Gram Schmidt (GS) para realizar la factorizacion
    retorna matrices Q y R calculadas con Gram Schmidt (y como tercer argumento opcional, el numero de operaciones)
    Si el metodo no esta entre las opciones, retorna None
    """

# Tests L05-QR:

import numpy as np

# --- Matrices de prueba ---
A2 = np.array([[1., 2.],
               [3., 4.]])

A3 = np.array([[1., 0., 1.],
               [0., 1., 1.],
               [1., 1., 0.]])

A4 = np.array([[2., 0., 1., 3.],
               [0., 1., 4., 1.],
               [1., 0., 2., 0.],
               [3., 1., 0., 2.]])

# --- Funciones auxiliares para los tests ---
def check_QR(Q,R,A,tol=1e-10):
    # Comprueba ortogonalidad y reconstrucción
    assert np.allclose(Q.T @ Q, np.eye(Q.shape[1]), atol=tol)
    assert np.allclose(Q @ R, A, atol=tol)

# --- TESTS PARA QR_by_GS2 ---
Q2,R2 = QR_con_GS(A2)
check_QR(Q2,R2,A2)

Q3,R3 = QR_con_GS(A3)
check_QR(Q3,R3,A3)

Q4,R4 = QR_con_GS(A4)
check_QR(Q4,R4,A4)

# --- TESTS PARA QR_by_HH ---
Q2h,R2h = QR_con_GS(A2)
check_QR(Q2h,R2h,A2)

Q3h,R3h = QR_con_HH(A3)
check_QR(Q3h,R3h,A3)

Q4h,R4h = QR_con_HH(A4)
check_QR(Q4h,R4h,A4)

# --- TESTS PARA calculaQR ---
Q2c,R2c = calculaQR(A2,metodo='RH')
check_QR(Q2c,R2c,A2)

Q3c,R3c = calculaQR(A3,metodo='GS')
check_QR(Q3c,R3c,A3)

Q4c,R4c = calculaQR(A4,metodo='RH')
check_QR(Q4c,R4c,A4)