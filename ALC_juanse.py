import numpy as np

from Labo06 import diagRH
from labo3 import multiplicar
from labo5_juanse import QR_con_HH

def obtenerQ3d(imgPts, objPts):
    """
    imgPts: (n,2) array con los (u,v) de cada punto en la imagen.
    objPts: (n,3) array con los (U,V,W) de cada punto en el mundo.
    Devuelve Q de tamaño (2n,12) según la ecuación 6.
    """
    n = imgPts.shape[0]
    if objPts.shape[0] != n:
        raise ValueError("imgPts y objPts deben tener la misma cantidad de puntos")

    Q = np.zeros((2*n, 12))
    for i in range(n):
        u, v = imgPts[i, 0], imgPts[i, 1]
        P = np.hstack([objPts[i], 1.0])   # punto homogéneo (U,V,W,1)
        Q[2*i, 0:4]    = P
        Q[2*i, 8:12]   = -u * P
        Q[2*i+1, 4:8]  = P
        Q[2*i+1, 8:12] = -v * P
    return Q

def resolverDLT3D(Q):
    """
    Q: (2n,12) matriz del sistema Qm = 0 (ecuación 6).
    Devuelve la matriz de proyección M (3,4).
    m es el autovector de QᵀQ de menor autovalor (equivale a la última columna de V en la SVD de Q).
    """
    #los devuelve ordenados de mayor a menor autovalor
    S, _ = diagRH(multiplicar(Q.T, Q))

    m = np.zeros(12)
    for i in range(12):
        m[i] = S[i][11]

    
    M = m.reshape(3, 4)

    return M

def devolverMatricesCalibracion3d(M, alpha_iguales = False, sin_sesgo = False):
    """
    M: (3,4) matriz de proyección, M = K [R | t].
    Devuelve R (3,3) rotación, t (3,) traslación y K (3,3) intrínsecos.
    """
    # Primeras 3 columnas: A = K R
    A = M[0:3 , 0:3]

    # A^-1 = R^T K^-1 = (ortogonal)(triangular superior), que es una QR
    A_inv = inversa_matriz(A)

    R_t, K_inv = QR_con_HH(A_inv)

    
    R = R_t.T

    K = inversa_matriz(K_inv)

    # el c de M se cancela acá (K_inv trae 1/c), así que t no hay que reescalarlo
    t = multiplicar(K_inv, M[:, 3])

    # M viene multiplicada por un c desconocido (m tiene norma 1), que queda metido en K.
    # Según la ec. 4, K[2,2] tiene que valer 1, así que dividimos por él para sacar c.
    K = K / K[2, 2]

    # Si m nos vino con el signo cambiado (c < 0), ese menos queda en R y t:
    # R sale como una reflexión (det = -1) y el objeto queda detrás de la cámara.
    # Dándolos vuelta a los dos la proyección no cambia y R pasa a ser una rotación.
    if determinante3x3(R) < 0:
        R = -R
        t = -t

    if alpha_iguales:
        promedio = (K[0][0] + K[1][1]) / 2 
        K[0][0] = promedio
        K[1][1] = promedio
    if sin_sesgo:
        K[0][1] = 0

    return R, t, K


def determinante3x3(A):
    """
    Calcula el determinante de una matriz A de 3x3 con la regla de Sarrus.
    """
    return (A[0, 0] * A[1, 1] * A[2, 2]
            + A[0, 1] * A[1, 2] * A[2, 0]
            + A[0, 2] * A[1, 0] * A[2, 1]
            - A[0, 2] * A[1, 1] * A[2, 0]
            - A[0, 0] * A[1, 2] * A[2, 1]
            - A[0, 1] * A[1, 0] * A[2, 2])



def inversa_matriz(A, tol=1e-12):
    """
    Calcula la inversa de una matriz cuadrada A (lista de listas).
    Lanza ValueError si la matriz no es cuadrada o es singular.
    """
    n = len(A)
    if n == 0 or any(len(fila) != n for fila in A):
        raise ValueError("La matriz debe ser cuadrada (n x n) para tener inversa.")
    # Construye la matriz aumentada [A | I]
    M = [[float(A[i][j]) for j in range(n)] +
         [1.0 if i == j else 0.0 for j in range(n)]
         for i in range(n)]
    for col in range(n):
        # Pivoteo parcial: busca la fila con mayor valor absoluto en la columna
        fila_max = max(range(col, n), key=lambda r: abs(M[r][col]))
        if abs(M[fila_max][col]) < tol:
            raise ValueError("La matriz es singular (no tiene inversa).")
        M[col], M[fila_max] = M[fila_max], M[col]
        # Normaliza la fila pivote
        pivote = M[col][col]
        M[col] = [x / pivote for x in M[col]]
        # Elimina la columna en las demás filas
        for r in range(n):
            if r != col and M[r][col] != 0:
                factor = M[r][col]
                M[r] = [a - factor * b for a, b in zip(M[r], M[col])]
    # La parte derecha de la matriz aumentada es la inversa
    return np.array([fila[n:] for fila in M])
