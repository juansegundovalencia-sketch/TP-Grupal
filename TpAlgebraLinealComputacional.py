#Tp algebra lineal computacional+
import numpy as np
from Labo8 import svd_reducida 
from tests_L05 import QR_con_GS
"""
Ejercicio 1 (DLT Ecuaciones formales). A partir de los datos compartidos en calib/3d armar el sistema de la ecuación 6. Contestar las siguientes preguntas:
a) Siendo Q ∈ R^2n×12, ¿qué rango debe tener para asegurar una solución no nula única a menos de un factor de escala? 
¿Cuántos puntos, como mínimo, aseguran la solución y qué condición deben cumplir?
Respuesta: para que halla solución única en Qm = 0, el rango de Q es 11, porque el nucleo tiene que tener dimension 1,se necesitas como mínimo 6 puntos 
porque para tener una única solución no nula el rango de Q debe ser 11, y al ser Q ∈ R^2n×12, n >= 6 para poder cumplir.


b) Encontrar el vector m que resuelve la minimización de la ecuación 8.
Respuesta:es un quilombo que explico con en el grupo

"""

"""
Ejercicio 2 (Calibración con puntos 3D - Implementación). Utilizando los datos de la carpetacalib/3d:
a) Implementar la función obtenerQ3d(imgPts, objPts) que devuelve la matriz Q a partir de la ecuación 6.
"""

#cv2.extrachesborcorner para obtener imgPts

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


if __name__ == "__main__":
    matriz = obtener_pixeles(" calib\3d\img_0001.jpg")
    print(f"Tamaño: {len(matriz[0])} x {len(matriz)}")
    print("Primer píxel:", matriz[0][0])  # Por ejemplo: (255, 0, 0, 255)


def obtenerQ3d(imgPts,objPts):
    Q = np.array((2,12))
    return 0


"""
b) A partir de lo resuelto en el Ejercicio 1 b), implementar la función resolverDLT3D(Q),
que a partir de Q encuentra la matriz de proyección M.
"""
def resolverDLT3D(Q):
    U, S, V = svd_reducida(Q)
    ultima_columna = V[:, -1]
    #Redimensiono el arreglo a una matriz de 3x4
    M = ultima_columna.reshape(3, 4)
    return M

    


"""
c) Una vez recuperada M, se la descompone en la matriz de parámetros intrínsecos K,
la matriz de rotación R y el vector de traslación t, imponiendo que R sea ortonormal
(RRT = I). Para ello, implementar la función devolverMatricesCalibracion3d(M),
que devuelva R, t y K.
"""
def devolverMatricesCalibracion3d(M):
    R = np.array((3,3))
    K = np.array((3,3))
    T = np.array(3)
    A = M[:, :3].copy() # copio las primeras columnas de M
    b = M[:, -1] # copio la ultima polumna

    # hago Descomposicion QR a A
    Q, R1 = QR_con_GS
    
    # R1 es ortogonal asi que R1^-1 = R^t 
    R = traspuesta(R1)
    # Obtengo K
    K = inversa_matriz(Q)
    # Obtengo T 
    T = productoMatriz(Q, b)

    return R, T, K


"""
d) Agregar a la función devolverMatricesCalibracion3d un parámetro opcional que permita imponer que αu = αv y/o K12 = 0.
"""



# Inversa de una matriz
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
    return [fila[n:] for fila in M]

# Transpuesta de una matriz
def traspuesta(A) :
    n_max,m_max = A.shape
    T = np.zeros((m_max,n_max))

    for n in range(0,n_max):
        for m in range(0,m_max):
            T[m][n] = A[n][m]

    return T

# Producto Matricial
def productoMatriz(A,B):
    m,n = A.shape
    mm,nn = B.shape
    if n != mm:
        print('No se pueden multiplicar')
        return None
    R = np.zeros((m,nn))
    for j in range(nn):
        R[:,j] = productoVector(A,B[:,j])
    return R

# Producto Matriz x Vector
def productoVector(A,v):
    m,n = A.shape
    nn = len(v)
    R = np.zeros(m)
    if n != nn:
        print('No se puede multiplicar')
        return None
    for i in range(m):
        R[i] = sum(A[i,:] * v)
    return(R)
