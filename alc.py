import numpy as np


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
    """Aplica una homografía a puntos 2D (n, 2) y devuelve (n, 2)."""
    raise NotImplementedError


def proyectarPuntosPinhole(K, R, t, objPts):
    """Proyección pinhole lambda [u v 1]^T = K [R t] [U V W 1]^T. Devuelve (n, 2)."""
    raise NotImplementedError

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
    """
    Resuelve A·b = 0 con ‖b‖ = 1, arma B = K⁻ᵀK⁻¹ y la factoriza con Cholesky para obtener K.

    Parámetros:
        A: matriz de 2N x 6 devuelta por armarA.

    Retorna:
        K: matriz intrínseca de 3 x 3 con K[2][2] = 1.
    """
    A = np.array(A, dtype=float)
    n = A.shape[1]

    # Iteración inversa: el autovector de menor autovalor de AᵀA domina (AᵀA + sI)⁻¹
    AtA = productoMatriz(traspuesta(A), A)
    inversaShift = inversa(AtA + 1e-9 * np.eye(n))
    b = np.ones(n)
    for _ in range(5):
        b = calcularAx(inversaShift, b)
        b = b / norma(b, 2)

    # Elegir el signo que hace B definida positiva (B₁₁ = 1/αᵤ² > 0)
    if b[0] < 0:
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

    r1 = lamda * producto_matriz_vector(K_inv, Hj_T[0, :])
    r2 = lamda * producto_matriz_vector(K_inv, Hj_T[1, :])
    r3 = productoVectorial(r1, r2)
    t =  lamda * producto_matriz_vector(K_inv, Hj_T[2, :])
    Rj = np.array([r1, r2, r3])

    return Rj, t

def proyectarARotacion(M):
    """Rotación ortogonal más cercana a M, vía SVD: R = U Vt, con det(R) = +1."""
    raise NotImplementedError


# ---------- Ejercicio 4: ubicación en la cancha ----------

def estimarHomografiaCancha(imgPts, canchaPts):
    """H (cancha -> imagen) a partir de correspondencias."""
    raise NotImplementedError


def imagenACancha(H, imgPts):
    """Mapea puntos de la imagen a la cancha usando inversa(H)."""
    raise NotImplementedError


def circuloEnCancha(centro, radio, n=100):
    """Muestrea n puntos de un círculo de radio dado en coordenadas de la cancha."""
    raise NotImplementedError


# ---------- Ejercicio 5: velocidad real ----------

def calibrarVistasSalguero(worldPts, imgPtsVistas):
    """Calibración a partir de las vistas de la esquina Salguero."""
    raise NotImplementedError


def velocidadPuntos(posiciones, tiempos):
    """Módulo de la velocidad entre frames consecutivos (n-1,)."""
    raise NotImplementedError


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
