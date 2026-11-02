import numpy as np

from labo3 import norma, multiplicar


### Funciones L05-QR
def QR_con_HH(A, tol=1e-12, extras=False):
    """
    A una matriz de m x n (m>=n)
    tol la tolerancia con la que se filtran elementos nulos en R
    extras : bool, opcional
        Si es True, devuelve tambien un diccionario con las matrices Q y R de cada paso
        (claves 'Q_matrices' y 'R_matrices'), para poder graficar el proceso.
    Devuelve la factorizacion QR de A usando reflectores de Householder:
        Q (m x m) ortogonal y R (m x n) triangular superior, con A = Q R.
    Si la matriz A no cumple m>=n, devuelve None
    """
    m, n = A.shape
    if m < n:
        return None

    # Arrancamos con R = A y Q = I, e iremos aplicando reflexiones: R <- H R, Q <- Q H
    R = np.array(A, dtype=float)
    Q = np.eye(m)

    R_matrices = []
    Q_matrices = []

    for k in range(n):
        # Columna k desde la diagonal para abajo: queremos llevarla a (alfa, 0, ..., 0)
        x = R[k:, k]
        norma_x = norma(x, 2)
        if norma_x < tol:
            continue  # la columna ya es nula debajo de la diagonal

        # Elegimos el signo de alfa opuesto al de x[0] para no restar numeros parecidos
        if x[0] >= 0:
            alfa = -norma_x
        else:
            alfa = norma_x

        # u = x - alfa*e1 es la normal al plano de reflexion que manda x a alfa*e1
        u = np.array(x, dtype=float)
        u[0] = u[0] - alfa
        norma_u = norma(u, 2)
        if norma_u < tol:
            continue  # x ya esta alineado con e1

        # Reflector de Householder del bloque: H_k = I - 2 u u^T / ||u||^2
        u = u.reshape(-1, 1)
        H_k = np.eye(m - k) - 2 * multiplicar(u, u.T) / norma_u**2

        # Lo embebemos en una identidad de m x m para no tocar las primeras k filas
        H = np.eye(m)
        H[k:, k:] = H_k

        # H es simetrica y ortogonal (H = H^T = H^-1)
        R = multiplicar(H, R)
        Q = multiplicar(Q, H)

        if extras:
            R_matrices.append(R.copy())
            Q_matrices.append(Q.copy())

    # Filtramos los valores casi nulos que quedan debajo de la diagonal por redondeo
    for i in range(m):
        for j in range(n):
            if i > j and abs(R[i, j]) < tol:
                R[i, j] = 0.0

    if extras:
        return Q, R, {'R_matrices': R_matrices, 'Q_matrices': Q_matrices}
    return Q, R


if __name__ == "__main__":
    # Tests L05-QR (los de Householder de tests_L05.py)
    A2 = np.array([[1., 2.],
                   [3., 4.]])

    A3 = np.array([[1., 0., 1.],
                   [0., 1., 1.],
                   [1., 1., 0.]])

    A4 = np.array([[2., 0., 1., 3.],
                   [0., 1., 4., 1.],
                   [1., 0., 2., 0.],
                   [3., 1., 0., 2.]])

    def check_QR(Q, R, A, tol=1e-10):
        # Comprueba ortogonalidad, reconstruccion y que R sea triangular superior
        assert np.allclose(Q.T @ Q, np.eye(Q.shape[1]), atol=tol)
        assert np.allclose(Q @ R, A, atol=tol)
        assert np.allclose(R, np.triu(R), atol=tol)

    for A in [A2, A3, A4]:
        Q, R = QR_con_HH(A)
        check_QR(Q, R, A)

    # Matrices rectangulares (m > n) al azar
    for _ in range(50):
        A = np.random.random((6, 4))
        Q, R = QR_con_HH(A)
        check_QR(Q, R, A)

    # Si m < n devuelve None
    assert QR_con_HH(np.random.random((2, 3))) is None

    print("Tests de QR_con_HH OK")
