"""
m과 n 크기를 가지는 행렬에 대해, 논문 곱셈식이 실제로 만족하는지를 확인하는 코드 - (6)
"""

from func import *
from diagonal_vector import *
import numpy as np

m = 8
n = 16
c = 8
j = 1

A = np.arange(1, n * m + 1).reshape(m , n)
B = np.arange(1, n * n +1).reshape(n, n)

print("=== A ===")
print(A)
print("=== B ===")
print(B)

# -------------------------------------------------------------

Ajl = np.zeros((n, n * c))
for l in range(n):
    for k in range(c):
        Ajl[l, k * n: (k+1) * n] = rotate(lower_diagonal_vector(A, c* j - l +k),l)

print("=== Ajl ===")
print(Ajl)

Bl = np.zeros((n, n * c))
for l in range(n):
    for k in range(c):
        Bl[l, k * n:(k+1)*n] = lower_diagonal_vector(B, l)

print("=== Bl ===")
print(Bl)

result = np.zeros(n*c)
for i in range(n):
    result += Ajl[i] * Bl[i]
    
print("=== result ===")
print(result)


print("\ncomparison")
for i in range(c):
    print(lower_diagonal_vector(np.dot(A,B), j * c + i))
