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


def QR_con_HH(A, tol=1e-12, extras=False):
    
    m, n = A.shape
    if m < n:
        return None

    R = np.array(A, dtype=float)
    Q = np.eye(m)

    R_matrices = []
    Q_matrices = []

    for k in range(n):
        x = R[k:, k]
        norma_x = norma(x, 2)
        if norma_x < tol:
            continue

        if x[0] >= 0:
            alfa = -norma_x
        else:
            alfa = norma_x

        u = np.array(x, dtype=float)
        u[0] = u[0] - alfa
        norma_u = norma(u, 2)
        if norma_u < tol:
            continue

        u = u.reshape(-1, 1)
        H_k = np.eye(m - k) - 2 * productoMatriz(u, u.T) / norma_u**2

        H = np.eye(m)
        H[k:, k:] = H_k

        R = productoMatriz(H, R)
        Q = productoMatriz(Q, H)

        if extras:
            R_matrices.append(R.copy())
            Q_matrices.append(Q.copy())

    for i in range(m):
        for j in range(n):
            if i > j and abs(R[i, j]) < tol:
                R[i, j] = 0.0

    if extras:
        return Q, R, {'R_matrices': R_matrices, 'Q_matrices': Q_matrices}
    return Q, R


def calculaQR(A, metodo='RH', tol=1e-12):
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

        S = productoMatriz(H, S_aux)

    for i in range(n):
        idx_max = i
        for j in range(i + 1, n):
            if D[j, j] > D[idx_max, idx_max]:
                idx_max = j
        if idx_max != i:
            D[i, i], D[idx_max, idx_max] = D[idx_max, idx_max], D[i, i]
            S[:, [i, idx_max]] = S[:, [idx_max, i]]

    return S, D


def svd_reducida(A, k="max", tol=1e-15):
    m_A, n_A = A.shape
    traspuesta = False
    if (m_A < n_A):
        A = A.T
        traspuesta = True

    S,D = diagRH(productoMatriz(A.T, A),tol,K =1000)

    umbral = tol * max(1, np.max(np.abs(np.diag(D))))

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

    if traspuesta:
        return Vñ, Sñ, Uñ.T
      
    return  Uñ.T ,Sñ, Vñ

def productoVectorial(a, b):
    return np.array([
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ])

def media(v):
    n = len(v)
    a = np.sum(v, axis=0) / n
    return a


def cholesky(B):
    B = np.array(B, dtype=float)
    n = B.shape[0]
    L = np.zeros((n, n))

    for j in range(n):
        suma = 0.0
        for k in range(j):
            suma += L[j][k] ** 2
        resto = B[j][j] - suma
        if resto <= 0:
            return None  # B no es definida positiva
        L[j][j] = np.sqrt(resto)

        for i in range(j + 1, n):
            suma = 0.0
            for k in range(j):
                suma += L[i][k] * L[j][k]
            L[i][j] = (B[i][j] - suma) / L[j][j]

    return L


def pseudoinversa(A):
    U, S, V = svd_reducida(A)
    r = len(S)
    Sinv = np.zeros((r, r))
    for i in range(r):
        Sinv[i, i] = 1.0 / S[i]
    return productoMatriz(productoMatriz(V, Sinv), U.T)

# =============================================================================
# 2. FUNCIONES DEL TRABAJO PRÁCTICO (esqueleto)
# =============================================================================

# ---------- Auxiliares de homografía y proyección ----------

def normalizarPuntos(pts):
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
    H = productoMatriz(productoMatriz(inversa(T_img),H_norm), T_world)
    return H

def obtenerHomografiaDLT(origPts, destPts):
    assert origPts.shape[0] >= 4 and destPts.shape[0] >= 4
    n = origPts.shape[0]

    origPts_norm, T_orig = normalizarPuntos(origPts)
    destPts_norm, T_dst = normalizarPuntos(destPts)

    A = np.zeros((2 * n, 9))
    for i in range(n):
        x, y = origPts_norm[i, 0], origPts_norm[i, 1]
        xp, yp = destPts_norm[i, 0], destPts_norm[i, 1]
        
        A[2 * i]     = [-x, -y, -1, 0, 0, 0, x * xp, y * xp, xp]
        A[2 * i + 1] = [0, 0, 0, -x, -y, -1, x * yp, y * yp, yp]
    n = origPts.shape[0]
    _, _, Vh = svd_reducida(A)

    H_norm = Vh[-1].reshape(3, 3)
    H = desnormalizarHomografia(H_norm, T_dst, T_orig)

    if H[2, 2] != 0:
        H = H / H[2, 2]
        
    return H

def proyectarPuntos(H, pts):
    n = len(pts)

    ptsMap = np.zeros((n, 2))
    for i in range(n):
        p = np.array([pts[i][0], pts[i][1], 1.0])
        q = producto_matriz_vector(H, p)

        ptsMap[i][0] = q[0] / q[2]
        ptsMap[i][1] = q[1] / q[2]

    return ptsMap


def proyectarPuntosPinhole(K, R, t, objPts):
    P = productoMatriz(K, np.hstack([R, np.array(t, dtype=float).reshape(3, 1)]))

    n = len(objPts)
    pts = np.zeros((n, 2))
    for i in range(n):
        q = producto_matriz_vector(P, np.hstack([np.array(objPts[i], dtype=float), 1.0]))
        pts[i][0] = q[0] / q[2]
        pts[i][1] = q[1] / q[2]
    return pts

from PIL import Image

def obtener_pixeles(ruta_imagen):
    with Image.open(ruta_imagen) as img:
        img = img.convert("RGBA")  # Unifica el formato para facilitar el manejo
        ancho, alto = img.size
        pixeles = list(img.getdata())

    # Organiza la lista plana en filas (alto x ancho)
    return [pixeles[i * ancho:(i + 1) * ancho] for i in range(alto)]




# ---------- Ejercicio 2: calibración con puntos 3D ----------

# 2.a)
def obtenerQ3d(imgPts, objPts):
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
    S, D = diagRH(productoMatriz(Q.T, Q))

    m = np.zeros(12)
    for i in range(12):
        m[i] = S[i][11]

    M = m.reshape(3, 4)

    return M


def devolverMatricesCalibracion3d(M, alpha_iguales=False, sin_sesgo=False):
    A = M[0:3 , 0:3]

    A_inv = inversa(A)

    R_t, K_inv = QR_con_HH(A_inv)

    R = R_t.T

    K = inversa(K_inv)

    t = calcularAx(K_inv, M[:, 3])

    K = K / K[2, 2]

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
    n = len(imgPts)
    Q = np.zeros((2 * n, 9))

    for i in range(n):
        U, V = objPts[i]
        u, v = imgPts[i]

        Q[2 * i]     = [U, V, 1, 0, 0, 0, -u * U, -u * V, -u]
        Q[2 * i + 1] = [0, 0, 0, U, V, 1, -v * U, -v * V, -v]

    return Q


def construirVij(ci, cj):
    return np.array([ci[0] * cj[0],
                     ci[0] * cj[1] + ci[1] * cj[0],
                     ci[1] * cj[1],
                     ci[2] * cj[0] + ci[0] * cj[2],
                     ci[2] * cj[1] + ci[1] * cj[2],
                     ci[2] * cj[2]])


def armarA(Hlst):
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
    U, S, V = svd_reducida(np.array(M, dtype=float))
    R = productoMatriz(U, V.T)
    if determinante_3x3(R) < 0:
        U[:, 2] = -U[:, 2]
        R = productoMatriz(U, V.T)
    return R


# ---------- Ejercicio 4: ubicación en la cancha ----------

def estimarHomografiaCancha(imgPts, canchaPts):

    Q = obtenerQ2d(imgPts, canchaPts)
    S , _ = diagRH(productoMatriz(Q.T, Q))
    
    h = np.zeros(9)
    
    for i in range(9):
        h[i] = S[i][8]

    return h.reshape(3,3)


def imagenACancha(H, imgPts):
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
    u0, v0 = centro_optico[0], centro_optico[1]

    # Con el centro óptico en el origen, K = diag(f, f, 1) y B = K^-T K^-1 = diag(w, w, 1), con w = 1/f^2.
    # Las dos condiciones de cada vista (r1 perpendicular a r2, y ||r1|| = ||r2||) quedan
    #   w (h11 h12 + h21 h22) + h31 h32 = 0
    #   w (h11^2 + h21^2 - h12^2 - h22^2) + (h31^2 - h32^2) = 0
    # que son de la forma a w + c = 0.
    a = []
    c = []
    for k in range(len(imgPtsVistas)):
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

    # cuadrados mínimos de a w + c = 0
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
    H = estimarHomografiaCancha(imgPts, np.array(worldPts, dtype=float)[:, 0:2])
    R, t = obtenerPose2D(H, K)
    # Si K no es del todo consistente con esta foto, r1 y r2 no salen del mismo largo ni
    # perpendiculares, y R no es una rotación. Nos quedamos con la rotación más cercana.
    R = proyectarARotacion(R)
    return R, t


# 5.c
def tiempoDesdeNombre(nombre_archivo):
    # Nos quedamos con el nombre sin carpetas ni extensión y lo partimos por los "_"
    nombre = nombre_archivo.split("/")[-1].split(".")[0]
    partes = nombre.split("_")
    h, m, s, ms = int(partes[1]), int(partes[2]), int(partes[3]), int(partes[4])
    return h * 3600 + m * 60 + s + ms / 1000


def velocidadPuntos(posiciones, tiempos):
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

def productoInterno(a, b):
    suma = 0.0
    for i in range(len(a)):
        suma += a[i] * b[i]
    return suma


def calculoLambdaMu(I0, J0):
    # (ecuación 17)
    re = productoInterno(J0, J0) - productoInterno(I0, I0)   # lambda^2 - mu^2
    im = -2.0 * productoInterno(I0, J0)                      # 2 lambda mu
    t = np.sqrt(re**2 + im**2)              # Módulo de C^2, que es igual a lambda^2 + mu^2
    # Despejamos ambas ecuaciones
    lam = np.sqrt(max((t + re) / 2, 0.0))
    mu = np.sqrt(max((t - re) / 2, 0.0))
    if im < 0:
        mu = -mu                        # lambda*mu tiene que tener el signo de im
    
    return [(lam, mu), (-lam, -mu)]     # Las dos raíces de C^2 son C y -C


def calcularSalidaPOSIT(Is, Js, fx, m0, M0):
    nI = norma(Is, 2)
    nJ = norma(Js, 2)
    Z0 = fx / ((nI + nJ) / 2)          # ||I|| = ||J||
    i = Is / nI
    j = Js / nJ
    k = productoVectorial(i, j)
    R = np.array([i, j, k])
    T = (Z0 / fx) * np.array([m0[0], m0[1], fx])   # vector O M0 en coords de cámara
    t = T - producto_matriz_vector(R, M0)
    return i, j, k, Z0, R, t


def evaluarPose(R, t, obj_vecs, img_vecs, K):
    obj_vecs = np.array(obj_vecs)
    img_vecs = np.array(img_vecs)
    n = obj_vecs.shape[0]
    validas = 0
    for p in range(n):
        Zp = producto_matriz_vector(R, obj_vecs[p])[2] + t[2]
        if Zp > 0:
            validas += 1
    proy = proyectarPuntosPinhole(K, R, t, obj_vecs)
    err2 = np.sum((proy - img_vecs) ** 2, axis=1)
    rms = np.sqrt(np.sum(err2) / n)
    return validas, rms


def aplicarPOSIT(obj_vecs, img_vecs, K, MaxIt, atol):
    obj_vecs = np.array(obj_vecs)
    img_vecs = np.array(img_vecs)
    fx, fy = K[0, 0], K[1, 1]
    cu, cv = K[0, 2], K[1, 2]

    x = img_vecs[:, 0] - cu
    y = (img_vecs[:, 1] - cv) * fx / fy
    x0, y0 = x[0], y[0]

    # Matriz A: vectores a_i = M0 Mi (i = 1..n-1)
    M0 = obj_vecs[0]
    A = obj_vecs[1:] - M0
    Ap = pseudoinversa(A)

    mejor = 0.0
    nvec = None
    for a in range(len(A)):
        for b in range(a + 1, len(A)):
            c = productoVectorial(A[a], A[b])
            if norma(c, 2) > mejor:
                mejor = norma(c, 2)
                nvec = c
    nvec = nvec / norma(nvec, 2)

    eps = np.zeros(len(A))
    rms_prev = float('inf')
    R_mejor, t_mejor = None, None

    for it in range(MaxIt):
        bu = x[1:] * (1 + eps) - x0
        bv = y[1:] * (1 + eps) - y0
        I0 = producto_matriz_vector(Ap, bu)
        J0 = producto_matriz_vector(Ap, bv)
        candidatas = []
        for lam, mu in calculoLambdaMu(I0, J0):
            I = I0 + lam * nvec
            J = J0 + mu * nvec
            i_, j_, k_, Z0, R, t = calcularSalidaPOSIT(I, J, fx, (x0, y0), M0)
            validas, rms = evaluarPose(R, t, obj_vecs, img_vecs, K)
            if validas == len(obj_vecs):          # descartamos poses con puntos detrás de la cámara
                candidatas.append((rms, R, t, k_, Z0))
        if len(candidatas) == 0:
            break
        rms, R, t, k_, Z0 = min(candidatas, key=lambda c: c[0])
        R_mejor, t_mejor = R, t
        eps = producto_matriz_vector(A, k_) / Z0
        if abs(rms_prev - rms) < atol:
            break
        rms_prev = rms

    return R_mejor, t_mejor
