"""

THOR:Secure Inference with Homomorphic Encryption 
3.1 ~ 3.2 Upper-lower Matrix Multiplication / Lower-lower Matrix Multiplication test

"""

import numpy as np
from func import rotate
from diagonal_vector import lower_diagonal_vector, upper_diagonal_vector, upper_diagonal_vector_prime, lower_diagonal_vector_prime


"""

upper-lower matrix multiplication test - proposition 1
A의 크기 nxm, B의 크기 mxn, n <= m 일 때
n은 m으로 나누어 떨어져야 한다.

"""

def upper_lower_mult_1(A, B, r):
    """
    A, B: input matrix
    r: r-th lower diagonal vector
    """

    n, m = A.shape
    Lr_AB = np.zeros(n)

    for l in range(m):
        L_B = lower_diagonal_vector(B, l)
        # print(L_B)
        U_A = upper_diagonal_vector(A, (l-r) % m)
        U_A = rotate(U_A, r)
        # print(U_A)
        Lr_AB += U_A * L_B
    
    return Lr_AB

"""

Upper-lower matrix multiplication test - proposition2
A is nxm, B is nxn, n>=m
n is divisible by m

"""

def upper_lower_mult_2(A, B, r):
    """
    A, B: input matrix
    r: r-th lower diagonal vector
    """

    _, n = A.shape
    Lr_AB = np.zeros(n)

    for l in range(n):
        L_B = lower_diagonal_vector(B, l)
        # print(L_B)
        U_A = upper_diagonal_vector_prime(A, l-r)
        U_A = rotate(U_A, r)
        # print(U_A)
        Lr_AB += U_A * L_B
    
    return Lr_AB


"""

lower-lower matrix multiplication test - corollary 1

"""

def lower_lower_mult1(A, B, r):
    
    n, m = A.shape
    Lr_AB = np.zeros(n)

    for l in range(m):
        L_B = lower_diagonal_vector(B, l)
        L_A = lower_diagonal_vector_prime(A, (n-l+r)%n)
        L_A = rotate(L_A, l)
        Lr_AB += L_A * L_B
    
    return Lr_AB


"""

lower-lower matrix multiplication test - corollary 2

"""

def lower_lower_mult2(A, B, r):

    m, n = A.shape
    Lr_AB = np.zeros(n)

    for l in range(n):
        L_B = lower_diagonal_vector(B, l)
        L_A = lower_diagonal_vector(A, (m-l+r)%m)
        L_A = rotate(L_A, l)
        Lr_AB += L_A * L_B
    
    return Lr_AB

if __name__ == "__main__":

    print("Upper-lower matrix multiplication test - proposition 1")
    n = 8
    m = 2
    A = np.arange(1, n*m+1).reshape(n, m)
    B = np.arange(1, n*m+1).reshape(m, n)
    print(np.dot(A,B))
    print()
    for i in range(n):
        print(upper_lower_mult_1(A, B, i))

    print("\n\nUpper-lower matrix multiplication test - proposition 2")
    n = 8
    m = 2
    A = np.arange(1, n*n+1).reshape(n, n)
    B = np.arange(1, n*m+1).reshape(n, m)
    print(np.dot(A, B))
    print()
    for i in range(m):
        print(upper_lower_mult_2(A, B, i))