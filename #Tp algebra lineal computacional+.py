#Tp algebra lineal computacional+
import numpy as np

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
def obtenerQ3d(imgPts,objPts):
    n
    Q = np.array((2*n,12))

    return 0


"""
b) A partir de lo resuelto en el Ejercicio 1 b), implementar la función resolverDLT3D(Q),
que a partir de Q encuentra la matriz de proyección M.
"""

"""
c) Una vez recuperada M, se la descompone en la matriz de parámetros intrínsecos K,
la matriz de rotación R y el vector de traslación t, imponiendo que R sea ortonormal
(RRT = I). Para ello, implementar la función devolverMatricesCalibracion3d(M),
que devuelva R, t y K.
"""

"""
d) Agregar a la función devolverMatricesCalibracion3d un parámetro opcional que permita imponer que αu = αv y/o K12 = 0.
"""

