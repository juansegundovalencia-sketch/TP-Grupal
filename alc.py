import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# =============================================================================
# 1. MÓDULO ALC DE LOS LABORATORIOS
# =============================================================================

def esCuadrada(A):
    A = np.array(A)
    return A.shape[0] == A.shape[1]


def triangSup(A):
    A = np.array(A)
    m, n = A.shape
    res = np.zeros((m, n))
    for i in range(m):
        for j in range(i, n):
            res[i][j] = A[i][j]
    return res


def triangInf(A):
    A = np.array(A)
    m, n = A.shape
    res = np.zeros((m, n))
    for i in range(m):
        for j in range(0, min(i + 1, n)):
            res[i][j] = A[i][j]
    return res


def diagonal(A):
    m, n = np.shape(A)
    res = np.zeros((m, n))
    for i in range(min(m, n)):
        res[i][i] = A[i][i]
    return res

def determinante_3x3(m):
  #Calcula el determinante de una matriz 3x3 

  # Extraemos los elementos fila por fila
  a, b, c = m[0]
  d, e, f = m[1]
  g, h, i = m[2]

  # Aplicamos la fórmula del determinante para 3x3
  det = a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g)

  return det

def traza(A):
    m, n = np.shape(A)
    res = 0
    for i in range(min(m, n)):
        res += A[i][i]
    return res


def traspuesta(A):
    if A is None:
        return None
    m, n = np.shape(A)
    trasp = np.zeros((n, m))
    for i in range(m):
        for j in range(n):
            trasp[j][i] = A[i][j]
    return trasp


def esSimetrica(A):
    if not esCuadrada(A):
        return False
    A = np.array(A)
    n = A.shape[0]
    for i in range(n):
        for j in range(i + 1, n):
            if A[i][j] != A[j][i]:
                return False
    return True


def productoMatriz(A, B):
    A = np.array(A, dtype=float)
    B = np.array(B, dtype=float)
    if A.ndim == 1:
        A = A.reshape(1, -1)
    if B.ndim == 1:
        B = B.reshape(-1, 1)
    m, p = A.shape
    p2, n = B.shape
    assert p == p2
    res = np.zeros((m, n))
    for i in range(m):
        for j in range(n):
            acum = 0.0
            for k in range(p):
                acum += A[i][k] * B[k][j]
            res[i][j] = acum
    return res


def calcularAx(A, x):
    A = np.array(A)
    m, n = A.shape
    b = np.zeros(m)
    for i in range(m):
        acum = 0
        for j in range(n):
            acum += A[i][j] * x[j]
        b[i] = acum
    return b


def intercambiarFilas(A, i, j):
    A[[i, j]] = A[[j, i]]
    return A


def sumar_fila_multiplo(A, i, j, s):
    m, n = np.shape(A)
    for k in range(n):
        A[i][k] += s * A[j][k]
    return A


def esDiagonalmenteDominante(A):
    A = np.array(A)
    m, n = A.shape
    for i in range(m):
        suma = 0
        for j in range(n):
            suma += abs(A[i][j])
        if suma >= 2 * abs(A[i][i]):
            return False
    return True


def matrizCirculante(v):
    u = np.array(v)
    n = u.shape[0]
    res = np.zeros((n, n))
    for x in range(n):
        res[0][x] = u[x]
    k = 1
    for i in range(1, n):
        for j in range(0, k):
            res[i][j] = u[n - k + j]
        for j in range(k, n):
            res[i][j] = u[j - k]
        k += 1
    return res


def matrizVandermonde(v):
    u = np.array(v)
    n = u.shape[0]
    res = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            res[i][j] = u[j] ** i
    return res


def fibonacci(n):
    if n == 0:
        return 0
    elif n == 1:
        return 1
    return fibonacci(n - 1) + fibonacci(n - 2)


def numeroAureo(n):
    return fibonacci(n + 1) / fibonacci(n)


def matrizFibonacci(n):
    res = np.zeros((n, n))
    for i in range(n):
        res[i][i] = fibonacci(i + i)
        for j in range(i + 1, n):
            res[i][j] = fibonacci(i + j)
            res[j][i] = fibonacci(i + j)
    return res


def matrizHilbert(n):
    res = np.zeros((n, n))
    for i in range(n):
        res[i][i] = 1 / (i + i + 1)
        for j in range(i + 1, n):
            res[i][j] = 1 / (i + j + 1)
            res[j][i] = 1 / (i + j + 1)
    return res


def norma(x, p):
    if p == 'inf':
        return np.max(np.abs(x))
    acum = 0
    for i in x:
        acum += np.abs(i) ** p
    return acum ** (1 / p)


def normaliza(X, p):
    Y = []
    for x in X:
        Y.append(x / norma(x, p))
    return Y


def normaMatMC(A, q, p, Np):
    A = np.array(A)
    n = np.shape(A)[1]
    vectores = np.random.randn(Np, n)
    v1 = normaliza(vectores, p)
    v2 = productoMatriz(np.array(v1), A.T)
    maximo = 0
    y = None
    for i in range(Np):
        aux = norma(v2[i], q)
        if aux > maximo:
            maximo = aux
            y = v1[i]
    return maximo, y


def normauno(A):
    A = np.array(A)
    m, n = np.shape(A)
    res = 0
    for j in range(n):
        actual = 0
        for i in range(m):
            actual += abs(A[i][j])
        if actual > res:
            res = actual
    return res


def normainfinito(A):
    A = np.array(A)
    m, n = np.shape(A)
    res = 0
    for i in range(m):
        actual = 0
        for j in range(n):
            actual += abs(A[i][j])
        if actual > res:
            res = actual
    return res


def normaExacta(A, p=[1, 'inf']):
    res = []
    if isinstance(p, int):
        if p == 1:
            res.append(normauno(A))
        else:
            return None
    elif isinstance(p, str):
        res.append(normainfinito(A))
    else:
        if p[0] != 1:
            return None
        res.append(normauno(A))
        res.append(normainfinito(A))
    return res


def condMC(A, p):
    A_inv = inversa(A)
    return normaMatMC(A, p, p, 1000)[0] * normaMatMC(A_inv, p, p, 1000)[0]


def condExacto(A, p):
    A_inv = inversa(A)
    normas_A = normaExacta(A, p)
    normas_inv = normaExacta(A_inv, p)
    return [a * b for a, b in zip(normas_A, normas_inv)]


def calculaLU(A):
    cant_op = 0
    if A is None:
        return None, None, 0

    m, n = np.shape(A)
    Ac = np.array(A, dtype=float).copy()

    if m != n:
        return None, None, 0

    for i in range(n - 1):
        if Ac[i][i] == 0:
            return None, None, 0
        for j in range(i + 1, n):
            factor = Ac[j][i] / Ac[i][i]
            cant_op += 1
            Ac[j][i] = factor
            for k in range(i + 1, n):
                Ac[j][k] -= Ac[j][i] * Ac[i][k]
                cant_op += 2

    if Ac[n - 1][n - 1] == 0:
        return None, None, cant_op

    L = triangInf(Ac)
    for i in range(n):
        L[i][i] = 1
    U = triangSup(Ac)
    return L, U, cant_op


def res_tri_inf(L, b):
    m, n = np.shape(L)
    x = np.zeros(n)
    for i in range(m):
        suma = 0
        for j in range(i):
            suma += L[i][j] * x[j]
        x[i] = (b[i] - suma) / L[i][i]
    return x


def res_tri_sup(U, c):
    m, n = np.shape(U)
    x = np.zeros(n)
    for i in range(n):
        fila = n - 1 - i
        suma = 0
        for j in range(fila + 1, n):
            suma += U[fila][j] * x[j]
        x[fila] = (c[fila] - suma) / U[fila][fila]
    return x


def res_tri(L, b, inferior=True):
    if inferior:
        return res_tri_inf(L, b)
    return res_tri_sup(L, b)


def inversa(A):
    A = np.array(A)
    n = A.shape[0]
    L, U, _ = calculaLU(A)
    if L is None or U is None:
        return None
    Inv = np.zeros((n, n))
    for i in range(n):
        e_i = np.zeros(n)
        e_i[i] = 1
        y = res_tri(L, e_i, inferior=True)
        x = res_tri(U, y, inferior=False)
        Inv[:, i] = x
    return Inv


def calculaLDV(A):
    L, U, _ = calculaLU(A)
    Vt, D, _ = calculaLU(traspuesta(U))
    return L, D, traspuesta(Vt)


def esSDP(A, atol=1e-8):
    L, D, V = calculaLDV(A)
    if D is None:
        return False
    for i in range(D.shape[0]):
        if D[i][i] <= 0:
            return False
    return esSimetrica(A)


def QR_con_GS(A, tol=1e-12, retorna_nops=False):
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
        vector_a_normalizar = np.zeros(n)
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
    raise NotImplementedError("Labo 5: QR_con_GS pendiente")


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
        H_k = np.eye(m - k) - 2 * productoMatriz(u, u.T) / norma_u**2

        # Lo embebemos en una identidad de m x m para no tocar las primeras k filas
        H = np.eye(m)
        H[k:, k:] = H_k

        # H es simetrica y ortogonal (H = H^T = H^-1)
        R = productoMatriz(H, R)
        Q = productoMatriz(Q, H)

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


def calculaQR(A, metodo='RH', tol=1e-12):
    """
    Elige 'RH' (Householder) o 'GS' (Gram-Schmidt) para factorizar A = Q R.
    Si el metodo no esta entre las opciones, devuelve None.
    """
    if metodo == 'RH':
        return QR_con_HH(A, tol)
    if metodo == 'GS':
        Q, R, _ = QR_con_GS(A, tol)
        return Q, R
    return None


def producto_matriz_vector(A, v):
    m, n = A.shape
    w = np.zeros(m)
    for i in range(m):
        for j in range(n):
            w[i] += A[i, j] * v[j]
    return w


def f_A(A, v):
    w_prime = producto_matriz_vector(A, v)
    n = w_prime.shape[0]
    acum = 0
    for i in range(n):
        acum += w_prime[i] ** 2
    norma_w = acum ** 0.5
    if norma_w > 0:
        return w_prime / norma_w
    return np.zeros(A.shape[0])


def f_k_A(A, k, v):
    A = np.array(A)
    w = np.array(v, dtype=float)
    for _ in range(k):
        w = f_A(A, w)
    return w


def metpot2k(A, tol=1e-15, K=1000):
    n = A.shape[0]
    v = np.random.rand(n)
    v_nuevo = f_k_A(A, 2, v)
    e = 0
    for i in range(n):
        e += v_nuevo[i] * v[i]
    k = 0

    while np.abs(e - 1) > tol and k < K:
        v = v_nuevo
        v_nuevo = f_k_A(A, 2, v)
        e = 0
        for i in range(n):
            e += v_nuevo[i] * v[i]
        k += 1

    autoval = 0
    u = producto_matriz_vector(A, v_nuevo)
    for i in range(n):
        autoval += v_nuevo[i] * u[i]
    eps = e - 1
    return v, autoval, k, eps


def diagRH(A, tol=1e-15, K=1e5):
    n = A.shape[0]

    v1, lam, _, _ = metpot2k(A, tol, K)

    e1 = np.zeros(n)
    e1[0] = 1

    u = e1 - v1

    if norma(u, 2) < tol:
        H =  np.eye(n)
    else:
        u = u.reshape(n, 1)
        H = np.eye(n) - 2 * (productoMatriz(u, u.T)) / norma(u, 2)**2

    #queremos ir obteniendo los primeros autovalores y autovectores sucesivamente
    if n == 2:
        S = H 
        D = productoMatriz(productoMatriz(H, A), traspuesta(H)) # conserva autoalroes.
    else:
        B = productoMatriz(productoMatriz(H, A), H.T)
        A_tilde = B[1:, 1:]

        S_tilde, D_tilde = diagRH(A_tilde, tol, K)

        D = np.zeros((n,n))
        D[0,0] = lam
        D[1:, 1:] = D_tilde

        S_aux = np.eye(n)
        S_aux[1:, 1:] = S_tilde

        S = productoMatriz(H, S_aux) # S aux solo tiene los autovectores de la sub matriz, multiplciando por H obtenemoes el autovector correspondiente a A-

    # Ordenamos autovalores de mayor a menor (y sus autovectores con ellos),
    # para no depender de que el método de la potencia haya convergido en cada paso
    for i in range(n):
        idx_max = i
        for j in range(i + 1, n):
            if D[j, j] > D[idx_max, idx_max]:
                idx_max = j
        if idx_max != i:
            D[i, i], D[idx_max, idx_max] = D[idx_max, idx_max], D[i, i]
            S[:, [i, idx_max]] = S[:, [idx_max, i]]

    return S, D


# ---------- Labo 8: SVD (PENDIENTE - implementar) ----------

def svd_reducida(A, k="max", tol=1e-15):
    m_A, n_A = A.shape
    transpuesta = False
    if (m_A < n_A):
        A = A.T
        transpuesta = True

    S,D = diagRH(productoMatriz(A.T, A),tol,K =1000)

    umbral = tol * max(1, np.max(np.abs(np.diag(D))))
    
    # recorre y eja solo los q no tengan autovalor cero

    n, m = D.shape
    
    Vñ = []
    Sñ = []
    Uñ = []

    contador = 0
    
    for i in range(n):
        if k != "max" and contador >= k:
            break
        if D[i][i] > umbral:
            contador += 1
            v = []
            for r in range(n): 
                v.append(S[r][i])
    
            Vñ.append(v)
            sigma = np.sqrt(D[i,i])
            Sñ.append(sigma)

            v = np.array(v)
            Uñ.append((producto_matriz_vector(A, v))/sigma)

    Vñ = np.array(Vñ)
    Uñ = np.array(Uñ)
    Vñ = Vñ.T

    if transpuesta:
        return Vñ, Sñ, Uñ.T
      
    return  Uñ.T ,Sñ, Vñ

def productoVectorial(a, b):
    """Producto vectorial a x b para vectores de R^3."""
    return np.array([
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ])

def media(v):
    n = v.len()
    a = np.sum(v) / n
    return a

# ---------- Pendientes del TP que no están en los labos ----------

def cholesky(B):
    """
    Factorización de Cholesky: B = L L^T, con L triangular inferior.
    B tiene que ser simétrica definida positiva; si no lo es, devuelve None.
    """
    B = np.array(B, dtype=float)
    n = B.shape[0]
    L = np.zeros((n, n))

    for j in range(n):
        # Diagonal: L[j,j]^2 = B[j,j] - (lo que ya aportaron las columnas anteriores)
        suma = 0.0
        for k in range(j):
            suma += L[j][k] ** 2
        resto = B[j][j] - suma
        if resto <= 0:
            return None  # B no es definida positiva
        L[j][j] = np.sqrt(resto)

        # Debajo de la diagonal: despejamos L[i,j] de B[i,j] = sum_k L[i,k] L[j,k]
        for i in range(j + 1, n):
            suma = 0.0
            for k in range(j):
                suma += L[i][k] * L[j][k]
            L[i][j] = (B[i][j] - suma) / L[j][j]

    return L


def pseudoinversa(A):
    """Pseudoinversa A+ a partir de svd_reducida."""
    raise NotImplementedError("Pendiente (usa svd_reducida)")

# =============================================================================
# 2. FUNCIONES DEL TRABAJO PRÁCTICO (esqueleto)
# =============================================================================

# ---------- Auxiliares de homografía y proyección ----------

def normalizarPuntos(pts):
    """
    Normalización isotrópica de Hartley para puntos (n, d).
    Devuelve (pts_norm, T) con pts_norm = T * pts en homogéneas.
    """
    media2 = media(pts)
    centrados = pts - media2
    distancia_media = media(np.sqrt(np.sum(centrados**2, axis=1)))
    
    escala = np.sqrt(2) / distancia_media
    
    # Matriz de transformación de similitud (T)
    T = np.array([
        [escala, 0, -escala * media2[0]],
        [0, escala, -escala * media2[1]],
        [0,           0,         1]])
    
    # Aplicar transformación a coordenadas homogéneas
    ones = np.ones((pts.shape[0], 1))
    pts_h = np.hstack([pts, ones])
    pts_norm_h = productoMatriz(T,pts_h.T).T
    
    return pts_norm_h[:, :2], T


def desnormalizarHomografia(H_norm, T_img, T_world):
    """H = inv(T_img) @ H_norm @ T_world (ecuación 14), usando productoMatricial."""
    H = productoMatriz(productoMatriz(inversa(T_img),H_norm), T_world)
    return H

def obtenerHomografiaDLT(origPts, destPts):
    """Homografía H (3x3) tal que dest ~ H orig, por DLT."""
    assert origPts.shape[0] >= 4 and destPts.shape[0] >= 4
    n = origPts.shape[0]

    # Normalizo los puntos (vital para la estabilidad numérica del SVD)
    origPts_norm, T_orig = normalizarPuntos(origPts)
    destPts_norm, T_dst = normalizarPuntos(destPts)

    # Construimos la matriz A (tamaño 2N x 9)
    A = np.zeros((2 * n, 9))
    for i in range(n):
        x, y = origPts_norm[i, 0], origPts_norm[i, 1]
        xp, yp = destPts_norm[i, 0], destPts_norm[i, 1]
        
        A[2 * i]     = [-x, -y, -1, 0, 0, 0, x * xp, y * xp, xp]
        A[2 * i + 1] = [0, 0, 0, -x, -y, -1, x * yp, y * yp, yp]
    n = origPts.shape[0]
    # Resolver el sistema homogéneo Ah = 0 usando SVD
    _, _, Vh = svd_reducida(A)

    # La solución h es la última fila de Vh (correspondiente al valor singular más pequeño)
    H_norm = Vh[-1].reshape(3, 3)

    # Desnormalizar la matriz obtenida: H = T_dst_inv @ H_norm @ T_src
    H = desnormalizarHomografia(H_norm, T_dst, T_orig)

    # Escalar para que H[2, 2] sea 1 (o estandarizar)
    if H[2, 2] != 0:
        H = H / H[2, 2]
        
    return H

def proyectarPuntos(H, pts):
    """
    Aplica una homografía a puntos 2D.
    H: (3,3) homografía (por ejemplo cancha -> imagen).
    pts: (n,2) array de puntos.
    Devuelve (n,2) array con los puntos transformados.
    """
    n = len(pts)

    ptsMap = np.zeros((n, 2))
    for i in range(n):
        # (x, y) -> (x, y, 1)
        p = np.array([pts[i][0], pts[i][1], 1.0])
        q = producto_matriz_vector(H, p)

        # divido por la tercera comp
        ptsMap[i][0] = q[0] / q[2]
        ptsMap[i][1] = q[1] / q[2]

    return ptsMap


def proyectarPuntosPinhole(K, R, t, objPts):
    """
    Proyección pinhole: lambda (u, v, 1) = K [R | t] (X, Y, Z, 1).
    K: (3,3) intrínsecos. R: (3,3) rotación. t: (3,) traslación.
    objPts: (n,3) array con los (X,Y,Z) de cada punto en el mundo.
    Devuelve (n,2) array con los (u,v) de cada punto en la imagen.
    """
    P = productoMatriz(K, np.hstack([R, np.array(t, dtype=float).reshape(3, 1)]))

    n = len(objPts)
    pts = np.zeros((n, 2))
    for i in range(n):
        # (X, Y, Z) -> (X, Y, Z, 1), proyectamos y dividimos por lambda (la tercera componente)
        q = producto_matriz_vector(P, np.hstack([np.array(objPts[i], dtype=float), 1.0]))
        pts[i][0] = q[0] / q[2]
        pts[i][1] = q[1] / q[2]
    return pts

#----------- Obtengo la matriz de pixeles --------
from PIL import Image

def obtener_pixeles(ruta_imagen):
    """
    Devuelve los píxeles de una imagen como una lista de filas.
    Cada píxel es una tupla (R, G, B) o (R, G, B, A) si la imagen tiene transparencia.
    """
    with Image.open(ruta_imagen) as img:
        img = img.convert("RGBA")  # Unifica el formato para facilitar el manejo
        ancho, alto = img.size
        pixeles = list(img.getdata())

    # Organiza la lista plana en filas (alto x ancho)
    return [pixeles[i * ancho:(i + 1) * ancho] for i in range(alto)]




# ---------- Ejercicio 2: calibración con puntos 3D ----------

# 2.a)
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

# 2.b)
def resolverDLT3D(Q):
    """
    Q: (2n,12) matriz del sistema Qm = 0 (ecuación 6).
    Devuelve la matriz de proyección M (3,4).
    m es el autovector de QᵀQ de menor autovalor (equivale a la última columna de V en la SVD de Q).
    """
    # Autovectores de QᵀQ (12x12); diagRH los devuelve ordenados de mayor a menor autovalor
    S, D = diagRH(productoMatriz(Q.T, Q))

    # El de menor autovalor es la última columna de S
    m = np.zeros(12)
    for i in range(12):
        m[i] = S[i][11]

    # m = [m1; m2; m3] son las filas de M apiladas, así que la armamos de 3 filas
    M = m.reshape(3, 4)

    return M


def devolverMatricesCalibracion3d(M, alpha_iguales=False, sin_sesgo=False):
    """
    M: (3,4) matriz de proyección, M = K [R | t].
    alpha_iguales: si es True, impone alpha_u = alpha_v (usando el promedio de los dos).
    sin_sesgo: si es True, impone K12 = 0.
    Devuelve R (3,3) rotación, t (3,) traslación y K (3,3) intrínsecos.
    """
    # Primeras 3 columnas: A = K R
    A = M[0:3 , 0:3]

    # A^-1 = R^T K^-1 = (ortogonal)(triangular superior), que es una QR
    A_inv = inversa(A)

    R_t, K_inv = QR_con_HH(A_inv)

    R = R_t.T

    K = inversa(K_inv)

    # el c de M se cancela acá (K_inv trae 1/c), así que t no hay que reescalarlo
    t = calcularAx(K_inv, M[:, 3])

    # M viene multiplicada por un c desconocido (m tiene norma 1), que queda metido en K.
    # Según la ec. 4, K[2,2] tiene que valer 1, así que dividimos por él para sacar c.
    K = K / K[2, 2]

    # Si m nos vino con el signo cambiado (c < 0), ese menos queda en R y t:
    # R sale como una reflexión (det = -1) y el objeto queda detrás de la cámara.
    # Dándolos vuelta a los dos la proyección no cambia y R pasa a ser una rotación.
    if determinante_3x3(R) < 0:
        R = -R
        t = -t

    if alpha_iguales:
        promedio = (K[0][0] + K[1][1]) / 2
        K[0][0] = promedio
        K[1][1] = promedio
    if sin_sesgo:
        K[0][1] = 0

    return R, t, K


# ---------- Ejercicio 3: calibración con puntos coplanares (Zhang) ----------

def obtenerQ2d(imgPts, objPts):
    """
    imgPts: (n,2) array con los (u,v) de cada esquina en la imagen.
    objPts: (n,2) array con los (U,V) de cada esquina en el tablero (W = 0).
    Devuelve Q de tamaño (2n,9) según la ecuación 11.
    """
    n = len(imgPts)
    Q = np.zeros((2 * n, 9))

    for i in range(n):
        U, V = objPts[i]
        u, v = imgPts[i]

        Q[2 * i]     = [U, V, 1, 0, 0, 0, -u * U, -u * V, -u]
        Q[2 * i + 1] = [0, 0, 0, U, V, 1, -v * U, -v * V, -v]

    return Q


def construirVij(ci, cj):
    """
    ci, cj: columnas i y j de una homografía H.
    Devuelve el vector v_ij (6,) tal que ci^T B cj = v_ij · b, con b = (B11, B12, B22, B13, B23, B33).
    """
    return np.array([ci[0] * cj[0],
                     ci[0] * cj[1] + ci[1] * cj[0],
                     ci[1] * cj[1],
                     ci[2] * cj[0] + ci[0] * cj[2],
                     ci[2] * cj[1] + ci[1] * cj[2],
                     ci[2] * cj[2]])


def armarA(Hlst):
    """
    Hlst: lista de las N homografías H_j (3,3), una por captura.
    Devuelve A (2N,6): cada H aporta 2 filas, r1 ⟂ r2 (v12) y ||r1|| = ||r2|| (v11 - v22).
    """
    N = len(Hlst)
    A = []
    for j in range(N):
        c1 = Hlst[j][:,0]
        c2 = Hlst[j][:,1]
        v11 = construirVij(c1,c1)
        v12 = construirVij(c1,c2)
        v22 = construirVij(c2,c2)

        A.append(v12)
        A.append(v11 - v22)

    return np.vstack(A)


def calcularMatrizK2D(A):

    A = np.array(A, dtype=float)
    n = A.shape[1]

    
    AtA = productoMatriz(traspuesta(A), A)
    S, _ = diagRH(AtA)
    b = np.zeros(n)
    for i in range(n):
        b[i] = S[i][n - 1]
    
    if b[0] < 0:  #todos los elementos de la diagonal tienen el mismo signo
        b = -b

    B = np.array([[b[0], b[1], b[3]],
                  [b[1], b[2], b[4]],
                  [b[3], b[4], b[5]]])

    # B = K⁻ᵀK⁻¹ = LLᵀ, con L = K⁻ᵀ, entonces K = (Lᵀ)⁻¹
    L = cholesky(B)
    K = inversa(traspuesta(L))

    return K / K[2][2]


def obtenerPose2D(Hj, K):
    """Pose (R_j, t_j) de una vista a partir de H_j y K (sección 2.2 h)."""
    Hj_T = Hj.T
    K_inv = inversa(K)
    lamda = 1 / norma(producto_matriz_vector(K_inv, Hj_T[0, :]), 2)

    # H puede venir con el signo cambiado (es solución de Qh = 0 igual que -H).
    # Si con este lamda el tablero queda detrás de la cámara (t3 < 0), usamos -lamda.
    t_prueba = lamda * producto_matriz_vector(K_inv, Hj_T[2, :])
    if t_prueba[2] < 0:
        lamda = -lamda

    r1 = lamda * producto_matriz_vector(K_inv, Hj_T[0, :])
    r2 = lamda * producto_matriz_vector(K_inv, Hj_T[1, :])
    r3 = productoVectorial(r1, r2)
    t =  lamda * producto_matriz_vector(K_inv, Hj_T[2, :])
    Rj = np.array([r1, r2, r3]).T

    return Rj, t

def proyectarARotacion(M):
    """
    Rotación más cercana a una matriz M de 3x3 (que "casi" es una rotación).
    Si M = U S V^T, la rotación más cercana es R = U V^T: es como dejar todos los
    valores singulares en 1. Si el determinante da -1 (una reflexión), damos vuelta
    la última columna de U para que quede +1.
    M: (3,3) array.
    Devuelve R (3,3) con R R^T = I y det(R) = +1.
    """
    U, S, V = svd_reducida(np.array(M, dtype=float))
    R = productoMatriz(U, V.T)
    if determinante_3x3(R) < 0:
        U[:, 2] = -U[:, 2]
        R = productoMatriz(U, V.T)
    return R


# ---------- Ejercicio 4: ubicación en la cancha ----------

def estimarHomografiaCancha(imgPts, canchaPts):
    """
    Estima la homografía H que lleva puntos de la cancha a la imagen.
    imgPts: (n,2) array con los (u,v) de cada punto en la foto, en píxeles.
    canchaPts: (n,2) array con los (U,V) de esos mismos puntos en la cancha, en metros.
    Hacen falta al menos 4 puntos y que no haya 3 alineados.
    Devuelve H (3,3), definida a menos de un factor de escala.
    h es el autovector de QᵀQ de menor autovalor, igual que m en resolverDLT3D.
    """

    Q = obtenerQ2d(imgPts, canchaPts)
    S , _ = diagRH(productoMatriz(Q.T, Q))
    
    h = np.zeros(9)
    
    for i in range(9):
        h[i] = S[i][8]

    return h.reshape(3,3)


def imagenACancha(H, imgPts):
    """
    Mapea puntos de la imagen a la cancha usando inversa(H)
    """
    n = len(imgPts)

    H_inv = inversa(H)

    ptsMap = np.zeros((n, 2))
    for i in range(n):
        # u,v,1
        p = np.array([imgPts[i][0], imgPts[i][1], 1.0])
        q = producto_matriz_vector(H_inv, p)

 	#divido por la tercera comp  
        ptsMap[i][0] = q[0] / q[2]
        ptsMap[i][1] = q[1] / q[2]

    return ptsMap



def circuloEnCancha(centro, radio, n=100):
    """
    Muestrea puntos de un círculo sobre el piso, en coordenadas de la cancha.
    centro: (U0, V0) en metros.
    radio: radio del círculo en metros.
    n: cantidad aproximada de puntos (la mitad para cada semicírculo).
    Devuelve (m,2) array con los (U,V) del círculo, cerrado (el último punto es el primero).
    Como no podemos usar np.cos ni np.sin, usamos la ecuación del círculo:
    (U - U0)^2 + (V - V0)^2 = r^2  =>  V = V0 ± sqrt(r^2 - (U - U0)^2)
    """
    U0, V0 = centro[0], centro[1]
    m = n // 2

    # U recorre de U0 - r a U0 + r en m pasos iguales
    paso = 2 * radio / (m - 1)
    dU = -radio + paso * np.arange(m)

    # lo que va adentro de la raíz; por redondeo puede dar un poquito negativo en los bordes
    adentro = radio**2 - dU**2
    adentro[adentro < 0] = 0
    dV = np.sqrt(adentro)

    # semicírculo de arriba de izquierda a derecha, y el de abajo de derecha a izquierda,
    # así los puntos quedan en orden y plt.plot dibuja el círculo sin cruzarse
    arriba = np.vstack([U0 + dU, V0 + dV]).T
    abajo = np.vstack([U0 + dU[::-1], V0 - dV[::-1]]).T

    # repetimos el primer punto al final para cerrar el dibujo
    return np.vstack([arriba, abajo, arriba[0:1]])


# ---------- Ejercicio 5: velocidad real ----------

# 5.a
def calibrarVistasSalguero(worldPts, imgPtsVistas, centro_optico):
    """
    Calcula K a partir de varias vistas de la esquina de Salguero, con el método del ejercicio 3
    pero simplificado: suponemos píxeles cuadrados (alpha_u = alpha_v = f), sin sesgo (K12 = 0)
    y el centro óptico (u0, v0) en el centro de la imagen. Así la única incógnita es f.
    worldPts: (n,3) array con los (X,Y,Z) de cada punto, en metros. Z se ignora: los puntos del
        cordón están a 0,1 m del piso y los tratamos como si estuvieran en Z = 0.
    imgPtsVistas: lista con un (n,2) array por vista, con los (u,v) de cada punto en esa foto.
    centro_optico: (u0, v0) en píxeles, el centro de la imagen.
    Devuelve K (3,3).
    """
    u0, v0 = centro_optico[0], centro_optico[1]

    # Con el centro óptico en el origen, K = diag(f, f, 1) y B = K^-T K^-1 = diag(w, w, 1), con w = 1/f^2.
    # Las dos condiciones de cada vista (r1 perpendicular a r2, y ||r1|| = ||r2||) quedan
    #   w (h11 h12 + h21 h22) + h31 h32 = 0
    #   w (h11^2 + h21^2 - h12^2 - h22^2) + (h31^2 - h32^2) = 0
    # que son de la forma a w + c = 0.
    a = []
    c = []
    for k in range(len(imgPtsVistas)):
        # Corremos los píxeles para que el centro óptico quede en (0, 0)
        pts = np.array(imgPtsVistas[k], dtype=float)
        pts_centrados = np.zeros(pts.shape)
        pts_centrados[:, 0] = pts[:, 0] - u0
        pts_centrados[:, 1] = pts[:, 1] - v0

        H = estimarHomografiaCancha(pts_centrados, np.array(worldPts, dtype=float)[:, 0:2])
        h1 = H[:, 0]
        h2 = H[:, 1]

        a.append(h1[0] * h2[0] + h1[1] * h2[1])
        c.append(h1[2] * h2[2])
        a.append(h1[0]**2 + h1[1]**2 - h2[0]**2 - h2[1]**2)
        c.append(h1[2]**2 - h2[2]**2)

    a = np.array(a)
    c = np.array(c)

    # Cuadrados mínimos para una sola incógnita: minimizar sum (a_i w + c_i)^2  =>  w = -sum(a c) / sum(a^2)
    w = -np.sum(a * c) / np.sum(a * a)
    if w <= 0:
        raise ValueError("w = 1/f^2 dio negativo: las vistas no alcanzan para estimar f")
    f = 1 / np.sqrt(w)

    K = np.array([[f, 0, u0],
                  [0, f, v0],
                  [0, 0, 1.0]])
    return K

# 5.b
def calcularPoseSalguero(worldPts, imgPts, K):
    """
    Calcula la pose de la cámara a partir de puntos del piso (Z = 0) y la K del 5a.
    Como todos los puntos son coplanares no sirve el DLT 3D (ver ejercicio 1a), así que
    hacemos como en el ejercicio 3: la homografía del piso y obtenerPose2D.
    worldPts: (n,3) array con los (X,Y,Z) de cada punto, en metros (todos con Z = 0).
    imgPts: (n,2) array con los (u,v) de cada punto en la imagen, en píxeles.
    K: (3,3) intrínsecos de la cámara.
    Devuelve R (3,3) y t (3,).
    """
    H = estimarHomografiaCancha(imgPts, np.array(worldPts, dtype=float)[:, 0:2])
    R, t = obtenerPose2D(H, K)

    # Si K no es del todo consistente con esta foto, r1 y r2 no salen del mismo largo ni
    # perpendiculares, y R no es una rotación. Nos quedamos con la rotación más cercana.
    R = proyectarARotacion(R)
    return R, t


# 5.c
def tiempoDesdeNombre(nombre_archivo):
    """
    Saca el instante de una captura a partir de su nombre, en segundos.
    Ej: 'img_10_18_47_351.jpg' son las 10:18:47 y 351 milisegundos.
    nombre_archivo: nombre (o ruta) del archivo.
    Devuelve el tiempo en segundos desde las 00:00.
    """
    # Nos quedamos con el nombre sin carpetas ni extensión y lo partimos por los "_"
    nombre = nombre_archivo.split("/")[-1].split(".")[0]
    partes = nombre.split("_")
    h, m, s, ms = int(partes[1]), int(partes[2]), int(partes[3]), int(partes[4])
    return h * 3600 + m * 60 + s + ms / 1000


def velocidadPuntos(posiciones, tiempos):
    """
    Velocidad de un agente entre capturas consecutivas.
    posiciones: (n,2) array con la posición (X,Y) del agente en el piso en cada captura, en metros.
    tiempos: (n,) array con el instante de cada captura, en segundos.
    Devuelve (n-1,) array con el módulo de la velocidad (m/s) entre cada captura y la siguiente.
    """
    n = len(posiciones)
    velocidades = np.zeros(n - 1)
    for i in range(n - 1):
        # distancia recorrida sobre el piso dividido el tiempo que pasó
        dx = posiciones[i + 1][0] - posiciones[i][0]
        dy = posiciones[i + 1][1] - posiciones[i][1]
        dt = tiempos[i + 1] - tiempos[i]
        velocidades[i] = np.sqrt(dx**2 + dy**2) / dt
    return velocidades


# ---------- Ejercicio 6: POSIT para puntos coplanares ----------

def calculoLambdaMu(I0, J0):
    """Lista de tuplas (lambda, mu) solución de la ecuación 16."""
    raise NotImplementedError


def calcularSalidaPOSIT(Is, Js, fx, m0, M0):
    """Pose posible (i, j, k, Z0, R, t) a partir de (I_s, J_s)."""
    raise NotImplementedError


def evaluarPose(R, t, obj_vecs, img_vecs, K):
    """Devuelve (cantidad de proyecciones válidas, RMS del error de reproyección)."""
    raise NotImplementedError


def aplicarPOSIT(obj_vecs, img_vecs, K, MaxIt, atol):
    """Ejecuta POSIT. Devuelve (R, t)."""
    raise NotImplementedError
