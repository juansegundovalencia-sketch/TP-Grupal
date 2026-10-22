#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 21 10:16:42 2026

@author: juan
"""
import numpy as np

import labo3


def fA(A,v): 
    wm = A@v
    norma2wm = labo3.norma(wm,2) 
    
    if( norma2wm > 0):
        return wm/ norma2wm
    else:
        return np.array([0.0] *len(v) )

def fAK(A, v , k):
    for i in range(k):
        v = fA(A, v)
    return v


def metpot2k(A, tol=1e-15, K= 1000):
    
    n = A.shape[0]
    
    v = np.random.rand(n)
    
    v = v/ labo3.norma(v, 2)
    
    vm = fAK(A,v, 2)
    
    #si apuntan en la misma dir, su prod escalar vale 1. Si es en direcciones opuestas -1, si es PERP 0.
    
    e = np.transpose(vm) @ v
    
    k = 0
    
    #usamos la tol para veer que tanto difiere
    while (np.abs(e-1) > tol and k < K):
        v = vm
        vm = fAK(A, v, 2)
        e = np.transpose(vm)@ v
        k = k + 1

    lam = np.transpose(vm) @ A @ v

    return vm, lam,k

def diagRH(A, tol= 1e-15, K= 1000):
    n = A.shape[0]

    v1,lam, _ = metpot2k(A, tol, K)

    e1 = np.zeros(n)
    e1[0] = 1

    u = e1 - v1

    if labo3.norma(u, 2) < tol:
        H =  np.eye(n)
    else:
        u = u.reshape(n, 1)
        H = np.eye(n) - 2 * (u @ u.T) / labo3.norma(u, 2)**2

    #queremos ir obteniendo los primeros autovalores y autovectores sucesivamente
    if n == 2:
        S = H 
        D = H @ A @ np.transpose(H) # conserva autoalroes.
    else:
        B = H @ A @ H.T
        A_tilde = B[1:, 1:]

        S_tilde, D_tilde = diagRH(A_tilde, tol, K)

        D = np.zeros((n,n))
        D[0,0] = lam
        D[1:, 1:] = D_tilde

        S_aux = np.eye(n)
        S_aux[1:, 1:] = S_tilde

        S = H @ S_aux # S aux solo tiene los autovectores de la sub matriz, multiplciando por H obtenemoes el autovector correspondiente a A-

    return S, D
#autovectores S