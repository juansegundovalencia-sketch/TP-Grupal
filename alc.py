
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
    """Factorización QR con reflectores de Householder. A de m x n con m >= n."""
    raise NotImplementedError("Labo 5: QR_con_HH pendiente")


def calculaQR(A, metodo='RH', tol=1e-12):
    """Elige 'RH' (Householder) o 'GS' (Gram-Schmidt)."""
    raise NotImplementedError("Labo 5: calculaQR pendiente")


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
        B = H @ A @ H.T
        A_tilde = B[1:, 1:]

        S_tilde, D_tilde = diagRH(A_tilde, tol, K)

        D = np.zeros((n,n))
        D[0,0] = lam
        D[1:, 1:] = D_tilde

        S_aux = np.eye(n)
        S_aux[1:, 1:] = S_tilde

        S = productoMatriz(H, S_aux) # S aux solo tiene los autovectores de la sub matriz, multiplciando por H obtenemoes el autovector correspondiente a A-

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

# ---------- Pendientes del TP que no están en los labos ----------

def cholesky(B):
    """Factorización B = L L^T con L triangular inferior, para B simétrica definida positiva."""
    raise NotImplementedError("Cholesky pendiente (lo usa el TP ej. 3)")


def pseudoinversa(A):
    """Pseudoinversa A+ a partir de svd_reducida."""
    raise NotImplementedError("Pendiente (usa svd_reducida)")


def productoVectorial(a, b):
    """Producto vectorial a x b para vectores de R^3."""
    return np.array([
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ])


# =============================================================================
# 2. FUNCIONES DEL TRABAJO PRÁCTICO (esqueleto)
# =============================================================================

# ---------- Auxiliares de homografía y proyección ----------

def normalizarPuntos(pts):
    """
    Normalización isotrópica de Hartley para puntos (n, d).
    Devuelve (pts_norm, T) con pts_norm = T * pts en homogéneas.
    """
    raise NotImplementedError


def desnormalizarHomografia(H_norm, T_img, T_world):
    """H = inv(T_img) @ H_norm @ T_world (ecuación 14), usando productoMatricial."""
    raise NotImplementedError


def obtenerHomografiaDLT(origPts, destPts):
    """Homografía H (3x3) tal que dest ~ H orig, por DLT."""
    raise NotImplementedError


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
    U, S, V = svd_reducida(Q)
    ultima_columna = V[:, -1]
    #Redimensiono el arreglo a una matriz de 3x4
    M = ultima_columna.reshape(3, 4)
    return M


# 2.c.d)
def devolverMatricesCalibracion3d(M, alpha_iguales=False, s_cero=False):
    R = np.array((3,3))
    K = np.array((3,3))
    T = np.array(3)
    A = M[:, :3].copy() # copio las primeras columnas de M
    b = M[:, -1] # copio la ultima polumna
    # hago Descomposicion QR a A
    Q, R1, _ = QR_con_GS(A)
    print("Q y R1")
    print(Q)
    print(R1)
    # R1 es ortogonal asi que R1^-1 = R^t 
    R = traspuesta(R1)
    # Obtengo K
    K = inversa(Q)
    
    # Aseguramos que los elementos de la diagonal de K sean positivos
    for i in range(3):
        if K[i, i] < 0:
            K[:, i] *= -1
            R[i, :] *= -1

    # Aplicamos las restricciones opcionales si se solicitan
    if s_cero:
        # K[0, 1] representa el sesgo o K_12 (s)
        K[0, 1] = 0

    if alpha_iguales:
        # Hacemos que alfa_u (K[0,0]) y alfa_v (K[1,1]) sean iguales (tomando el promedio)
        alpha_prom = (K[0, 0] + K[1, 1]) / 2.0
        K[0, 0] = alpha_prom
        K[1, 1] = alpha_prom

    # Si modificamos K, R puede perder la ortogonalidad estricta.
    # Re-ajustamos R utilizando SVD para asegurar que sea una matriz de rotación válida.
    if alpha_iguales or s_cero:
        R_temp = productoMatriz(inversa(K), A)
        Uñ, _, Vt = svd_reducida(R_temp)
        R2 = productoMatriz(Uñ, Vt)
        
        # Aseguramos que el determinante de R sea +1 (evitar inversión espacial)
        if determinante_3x3(R) < 0:
            Uñ[:, -1] *= -1
            R2 = productoMatriz(Uñ, Vt)

    # Normalizamos K para que su último elemento K[2,2] sea 1
    K = K / K[2, 2]

    # Obtengo el vector de traslación T a partir de b = K * T  =>  T = inv(K) * b
    T = productoMatriz(K, b)

    return R2, T, K

if __name__ == "__main__":
    matriz = obtener_pixeles(r"TP_template\ calib\3d\img_0001.jpg") # es mi ruta a la primera imagen de calib/3d
    print(f"Tamaño: {len(matriz[0])} x {len(matriz)}")
    print("Primer píxel:", matriz[0][0])  # Por ejemplo: (255, 0, 0, 255)
    base = r"TP_template\ calib\3d"
    corners = np.loadtxt(base + r"\detected_corners.txt")
    objPts  = np.loadtxt(base + r"\p_W_corners.txt", delimiter=",")
    imgPts  = corners[0].reshape(-1, 2)
    #print(obtenerQ3d(imgPts,objPts))= Q
    Q = obtenerQ3d(imgPts,objPts)
    #print(resolverDLT3D(Q)) HASTA ACA CONSEGUIMOS M => esto esta chequedo
    M = resolverDLT3D(Q)
    # revisar QR_conGS R no da ortogonal y Q es un numero cuando tiene que ser matriz
    print(devolverMatricesCalibracion3d(M, alpha_iguales=False, s_cero=False))

# ---------- Ejercicio 3: calibración con puntos coplanares (Zhang) ----------

def obtenerQ2d(imgPts, objPts):
    """Matriz Q (2n x 9) del sistema Q h = 0 para la homografía de una vista (ecuación 11)."""
    raise NotImplementedError


def armarA(Hlist):
    """Matriz A (2N x 6) a partir de las N homografías (sección 2.2 e)."""
    raise NotImplementedError


def calcularMatrizK2D(A):
    """Resuelve A b = 0, arma B, la factoriza con Cholesky y devuelve K."""
    raise NotImplementedError


def obtenerPose2D(Hj, K):
    """Pose (R_j, t_j) de una vista a partir de H_j y K (sección 2.2 h)."""
    raise NotImplementedError


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
