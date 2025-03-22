"""
rotation과 masking이 제대로 되고 있는지 확인하는 코드.
구현한 rotation과 masking 결과인 Ajl을 실제 값과 비교한다. - (5)
"""

import numpy as np
from func import *
from diagonal_vector import *
from masking_test import mask


c = 2
m = 8
n = 64

for j in range(m//c):
    for l in range(n):

        A = np.arange(1, m * n + 1).reshape(m, n)
        # print("-----A-----")
        # print(A)
        # print("\n")


        ctA = np.zeros((m//c, n*c))
        for i in range(m//c):
            for k in range(c):
                ctA[i, n*k: n*(k+1)] = lower_diagonal_vector(A, i*c+k)
        # print("------ctA------")
        # print(ctA)
        # print("\n")


        mul0, mul1, mul2, mul3 = mask(n, c, l)
        # print("mu0:", mul0)
        # print("mu1:", mul1)
        # print("mu2:", mul2)
        # print("mu3:", mul3)

        # print("\n")

        ctAj_before = ctA[(j + -1 * (l // c) - 1)%(m//c)]
        ctAj_now = ctA[(j + -1 * (l// c))% (m//c)]

        ct_before = rotate(ctAj_before, -n*(l%c)+l)
        ct_now = rotate(ctAj_now, -n*(l%c)+l)

        ct_before_prime = rotate(ctAj_before, -n*(l%c+1)+l)
        ct_now_prime = rotate(ctAj_now, -n*(l%c+1)+l)

        ctAjl = ct_before*mul0 + ct_now*mul1 + ct_before_prime*mul2 + ct_now_prime*mul3

        # print("ct_before:", ct_before)
        # print("ct_now:", ct_now)
        # print("ct_before_prime:", ct_before_prime)
        # print("ct_now_prime:", ct_now_prime)
        # print("\n")

        # print("computed ctAjl: ", ctAjl)
        # print("\n")

        real_ctAjl = np.zeros(c*n)
        for i in range(c):
            real_ctAjl[i*n:(i+1)*n] = rotate(lower_diagonal_vector(A, c*j-l+i), l)
        # print("real ctAjl:", real_ctAjl)

        if (np.array_equal(real_ctAjl, ctAjl)):
            print("pass")
        else:
            print("fail")

