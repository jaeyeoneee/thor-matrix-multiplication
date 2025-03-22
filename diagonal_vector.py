import numpy as np
from func import rotate, get_submatrix

"""
nxm matrix가 input으로 들어왔을 때,
n > m 이면 lower diagonal vector일 때 전체 벡터를 표현할 수 없다. 이때의 곱셈을 위해 prime 함수 정의.
n < m 이면 upper diagonal vector일 때 전체 벡터를 표현할 수 없다. 이때의 곱셈을 위해 prime 함수 정의.
"""


def upper_diagonal_vector(A, k):
    """
    A : a matrix 
    k : k-th upper diagonal vector
    """

    a, b = A.shape
    max_v = max(a, b)

    Uk = np.zeros(max_v)
    for t in range(max_v):
        i = t % a
        j = (k + t) % b
        Uk[t] = A[i, j]

    return Uk

def upper_diagonal_vector_prime(A, k):
    
    m, _ = A.shape

    if k < m:
        return upper_diagonal_vector(A, k)
    
    Uk_prime = upper_diagonal_vector(A, k % m)
    Uk_prime = rotate(Uk_prime, m*(k//m))

    return Uk_prime


def lower_diagonal_vector(A, k):
    """
    
    A : a matrix
    k : k-th lower diagonal vector

    """

    a, b = A.shape
    max_v = max(a, b)

    Lk = np.zeros(max_v)
    for t in range(max_v):
        i = (t + k) % a
        j = t % b
        Lk[t] = A[i, j]
    
    return Lk

def lower_diagonal_vector_prime(A, k):

    n, m = A.shape

    if k < m:
        return lower_diagonal_vector(A, k)
    
    Lk_prime = lower_diagonal_vector(A, k % m)
    Lk_prime = rotate(Lk_prime, m * (k//m))

    return Lk_prime


def interlaced_upper_diagonal_vector(A, k):
    """
    A: an array of sliced matrices 
    k: k-th upper diagonal vector

    row and colum size of a sliced matrix is the same.
    sliced matrix가 input으로 들어오면 그것의 upper diagonal vector를 return한다.
    """

    n_sliced = len(A)
    n, _ = A[0].shape

    dim = n_sliced * n
    Uk_interlaced = np.zeros(dim)
    for t in range(dim):
        matrix_order = t % n_sliced
        index = t // n_sliced
        Uk_interlaced[t] = upper_diagonal_vector(A[matrix_order], k)[index]

    return Uk_interlaced

def interlaced_lower_diagonal_vector(A, k):
    n_sliced = len(A)
    n, _ = A[0].shape

    dim = n_sliced * n
    Lk_interlaced = np.zeros(dim)
    for t in range(dim):
        matrix_order = t % n_sliced
        index = t // n_sliced
        Lk_interlaced[t] = lower_diagonal_vector(A[matrix_order], k)[index]
    return Lk_interlaced

if __name__ == "__main__":

    " 3x4 matrix example "
    A1 = np.array([
        [1, 2, 3, 4],
        [5, 6, 7, 8],
        [9, 10, 11, 12]
    ]) 

    " 4x4 matrix example "
    A2 = np.array([
        [1, 2, 3, 4],
        [5, 6, 7, 8],
        [9, 10, 11, 12],
        [13, 14, 15, 16]
    ])

    " 4x3 matrix example"
    A3 = np.array([
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9],
        [10, 11, 12]
    ])


    a, b = A1.shape

    print("----A1-----")
    print(A1)
    print()
    for i in range(min(a, b)):
        print("---upper diagonal vector with ", i, "----")
        print(upper_diagonal_vector(A1, i))
    print("n<m 일 때는 upper diagonal vector로 표현 불가\n\n")
    

    print("----A1-----")
    print(A1)
    print()
    for i in range(min(a, b)):
        print("---lower diagonal vector with ", i, "----")
        print(lower_diagonal_vector(A1, i))
    print("n<m일 때는 lower diagonal vector로 표현 가능\n\n")

    print("----A2-----")
    print(A2)
    a, b = A2.shape
    for i in range(min(a, b)):
        print("---upper diagonal vector with ", i, "----")
        print(upper_diagonal_vector(A2, i))
    print("\n\n")

    print("----A2-----")
    print(A2)
    for i in range(min(a, b)):
        print("---lower diagonal vector with ", i, "----")
        print(lower_diagonal_vector(A2, i))

    print("\n")
    print("----A3----")
    print(A3)
    a, b = A3.shape
    for i in range(min(a, b)):
        print("---upper diagonal vector with", i, "----")
        print(upper_diagonal_vector(A3, i))

    print("\n")
    print("----A3-----")
    print(A3  )
    for i in range(min(a, b)):
        print("----lower diagonal vector with", i, "----")
        print(lower_diagonal_vector(A3, i))

