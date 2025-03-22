import numpy as np

def rotate(A, k):
    """
    ckks에서 rotate는 왼쪽 방향으로 이루어짐.
    """

    return np.roll(A, -k)


import numpy as np

def get_submatrix(matrix, n, i, j):
    """
    Given an (a, b) matrix, divide it into (n, n) blocks and return the (i, j)-th block.
    
    Parameters:
    - matrix: np.ndarray of shape (a, b)
    - n: int, size of each submatrix (must be a divisor of both a and b)
    - i: int, row index of the desired submatrix
    - j: int, column index of the desired submatrix
    
    Returns:
    - np.ndarray of shape (n, n) corresponding to the (i, j) block.
    """
    a, b = matrix.shape
    
    if a % n != 0 or b % n != 0:
        raise ValueError("n must be a divisor of both a and b")

    num_rows = a // n
    num_cols = b // n
    
    if not (0 <= i < num_rows and 0 <= j < num_cols):
        raise IndexError("Index out of range for submatrices")

    row_start, row_end = i * n, (i + 1) * n
    col_start, col_end = j * n, (j + 1) * n
    
    return matrix[row_start:row_end, col_start:col_end]

def get_ith_matrices(A, i, d_num):
    """
    A: input matrix
    i: A^(i-th) matrix
    d_num: the number of diagonal matrices

    A matrix의 크기는 d x d 라고 가정
    d_num = d // n, 즉, d 크기를 n으로 자르고, 그때 d_num 개의 matrix로 나누어진다.
    return: A^(i) = (Ai,0, Ai+1,1, ..., Ai+d-1,d-1)
    """

    d, d = A.shape
    n = d // d_num
    Ai = np.zeros((d_num, n, n))
    for j in range(d_num):
        Ai[j] = get_submatrix(A, n, (i+j)%d_num, j%d_num)
    return Ai

# A = np.arange(1, 17).reshape(4, 4)
# print(A)
# print(get_ith_matrices(A, 1, 4//2))

# # Example usage
# a, b = 8, 8  # Example matrix size
# n = 4         # Submatrix size
# matrix = np.arange(a * b).reshape(a, b)  # Example matrix with values 0 to (a*b - 1)
# print(matrix)

# # Get the (1, 1) submatrix
# submatrix = get_submatrix(matrix, n, 1, 1)
# print(submatrix)

def powers_of_two_up_to_n(n):
    """
    n이 2의 거듭제곱수임을 가정하고,
    1부터 n까지 (포함) 가능한 모든 2의 거듭제곱을 담은 numpy 배열을 반환한다.
    예) n=16이면 [2,4,8,16]
    1은 제외.
    """
    import math
    max_exp = int(math.log2(n))  
    arr = [2**i for i in range(max_exp+1)]
    return np.array(arr)[1:]

