#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep  1 11:37:43 2026

@author: juan
"""

import numpy as np
import matplotlib.pyplot as plt


def norma(x, p):
    if p == 'inf':
        return np.max(np.abs(x))
    return np.sum(np.abs(x) ** p) ** (1 / p)


def normaliza(X, p):
    res = []
    for e in X:
        n = norma(e, p)
        if n == 0:
            res.append(np.array(e, dtype=float))
        else:
            res.append(e / n)
    return res


def graficar():
    p_valores = [1, 2, 5, 10, 100, 200]
    x = np.linspace(-1, 1, 1000)

    for p in p_valores:
        # Evitamos potencias de números negativos por errores numéricos.
        interior = np.maximum(0, 1 - np.abs(x) ** p)
        y = interior ** (1 / p)
        plt.plot(x, y, label=f"p={p}")
        plt.plot(x, -y)

    plt.axis("equal")
    plt.grid()
    plt.legend()
    plt.show()


def normaMatMC(A, q, p, Np):
    A = np.asarray(A, dtype=float)
    n = A.shape[1]

    if Np <= 0:
        raise ValueError("Np debe ser un entero positivo.")

    vectores = np.random.uniform(-1, 1, (Np, n))

    # En dimensión 2 incluimos algunas direcciones conocidas.
    # Solo las reemplazamos si hay suficientes muestras.
    if n == 2 and Np >= 6:
        vectores[0] = [1, 0]
        vectores[1] = [0, 1]
        vectores[2] = [1, 1]
        vectores[3] = [-1, 1]
        vectores[4] = [1, -1]
        vectores[5] = [-1, -1]

    maximo = -1.0
    x_max = None

    for x in vectores:
        norma_x = norma(x, p)
        if norma_x == 0:
            continue

        valor = norma(A @ x, q) / norma_x
        if valor > maximo:
            maximo = valor
            x_max = x.copy()

    return maximo, x_max

def normaExacta(A, p):
    A = np.asarray(A, dtype=float)
    filas, columnas = A.shape

    if p == 'inf':
        maximo = 0
        for i in range(filas):
            suma = 0
            for j in range(columnas):
                suma += np.abs(A[i][j])
            if suma > maximo:
                maximo = suma
        return maximo

    elif p == 1:
        maximo = 0
        for j in range(columnas):
            suma = 0
            for i in range(filas):
                suma += np.abs(A[i][j])
            if suma > maximo:
                maximo = suma
        return maximo

    else:
        return None


def condMC(A, p, Np=1000):
    norma_A = normaMatMC(A, p, p, Np)[0]
    A_inv = np.linalg.inv(A)
    norma_A_inv = normaMatMC(A_inv, p, p, Np)[0]
    return norma_A * norma_A_inv


# Prueba de condicionamiento por Monte Carlo.
# La comparación entre dos estimaciones aleatorias puede fallar por variación
# del muestreo, incluso si el código está bien. Por eso no la usamos como
# assert automático.
#
# A_cond = np.array([[3, 2], [4, 1]], dtype=float)
# normaA = normaMatMC(A_cond, 2, 2, 10000)
# A_inv = np.linalg.solve(A_cond, np.eye(A_cond.shape[0]))
# normaA_inv = normaMatMC(A_inv, 2, 2, 10000)
# condA = condMC(A_cond, 2, 10000)
# assert np.allclose(normaA[0] * normaA_inv[0], condA, atol=1e-3)


# Tests de normaExacta
assert np.allclose(normaExacta(np.array([[1, -1], [-1, -1]]), 1), 2)
assert np.allclose(normaExacta(np.array([[1, -2], [-3, -4]]), 1), 6)
assert np.allclose(normaExacta(np.array([[1, -2], [-3, -4]]), 'inf'), 7)
assert normaExacta(np.array([[1, -2], [-3, -4]]), 2) is None
assert normaExacta(np.random.random((10, 10)), 1) <= 10
assert normaExacta(np.random.random((4, 4)), 'inf') <= 4


# Tests de normaMatMC
nMC = normaMatMC(A=np.eye(2), q=2, p=1, Np=100000)
assert np.allclose(nMC[0], 1, atol=1e-3)
assert (
    np.allclose(np.abs(nMC[1][0]), 1, atol=1e-3)
    or np.allclose(np.abs(nMC[1][1]), 1, atol=1e-3)
)
assert (
    np.allclose(np.abs(nMC[1][0]), 0, atol=1e-3)
    or np.allclose(np.abs(nMC[1][1]), 0, atol=1e-3)
)

nMC = normaMatMC(A=np.eye(2), q=2, p='inf', Np=100000)
assert np.allclose(nMC[0], np.sqrt(2), atol=1e-3)
assert np.allclose(np.abs(nMC[1][0]), 1, atol=1e-3)
assert np.allclose(np.abs(nMC[1][1]), 1, atol=1e-3)

A = np.array([[1, 2], [3, 4]], dtype=float)
nMC = normaMatMC(A=A, q='inf', p='inf', Np=1000000)
# La estimación de Monte Carlo puede compararse visualmente con:
# normaExacta(A, 'inf')


# Tests de norma
assert np.allclose(norma(np.array([1, 1]), 2), np.sqrt(2))
assert np.allclose(norma(np.array([1] * 10), 2), np.sqrt(10))
assert norma(np.random.rand(10), 2) <= np.sqrt(10)
assert norma(np.random.rand(10), 2) >= 0

for x in normaliza([np.array([1] * k) for k in range(1, 11)], 2):
    assert np.allclose(norma(x, 2), 1)

for x in normaliza([np.array([1] * k) for k in range(2, 11)], 1):
    print(not np.allclose(norma(x, 2), 1))

for x in normaliza([np.random.rand(k) for k in range(1, 11)], 'inf'):
    assert np.allclose(norma(x, 'inf'), 1)

print("Todos los tests activos terminaron correctamente.")