"""
4.3.1 Packed Lower-lower Matrix Multiplication 

nxm matrix/mxn matrix multiplication
"""

import numpy as np
from func import *
from diagonal_vector import *

def mask(n, c, ell):
    s = n*c
    mu_l0 = np.zeros(s)
    mu_l1 = np.zeros(s)
    mu_l2 = np.zeros(s)
    mu_l3 = np.zeros(s)

    ell_mod_c = ell % c

    for i in range(s):
        r = i // n
        local_idx = i % n

        is_front = (local_idx < (n - ell))
        
        if (ell_mod_c <= r < c):
            if is_front:
                mu_l1[i] = 1.0
            else:
                mu_l3[i] = 1.0
        else:
            if is_front:
                mu_l0[i] = 1.0
            else:
                mu_l2[i] = 1.0

    return mu_l0, mu_l1, mu_l2, mu_l3


def cipher_cipher_matmul(A_cipher, B_cipher, n, m, c):

    n_tilde = n // c

    C_cipher = [None]*n_tilde
    
    ctC_jl = {}
    for j in range(n_tilde):
        c_init = A_cipher[(j)%(n//c)] * B_cipher[0]
        C_cipher[j] = c_init
        for ell in range(1, m):
            ctj_ell = rotate(A_cipher[(j + -1 * (ell // c))%(n//c)], -n * (ell % c) + ell)
            product = ctj_ell * B_cipher[ell]
            ctC_jl[(j, ell)] = product

    ct_part = {}
    for ell in range(1, m):
        mu_l0, mu_l1, mu_l2, _= mask(n, c, ell)
        for j in range(n_tilde):
            tmp = ctC_jl[(j, ell)]
            ctj_ell_0 = tmp * mu_l0
            ctj_ell_1 = tmp * mu_l1
            ctj_ell_2 = tmp * rotate(mu_l2, n)
            ctj_ell_3 = tmp - ctj_ell_0 - ctj_ell_1 - ctj_ell_2

            ct_part[(j, ell, 0)] = ctj_ell_0
            ct_part[(j, ell, 1)] = ctj_ell_1
            ct_part[(j, ell, 2)] = ctj_ell_2
            ct_part[(j, ell, 3)] = ctj_ell_3
    
    for j in range(n_tilde):
        ctCPrime_j = np.zeros_like(C_cipher[0])
        CtCDPrime_j = np.zeros_like(C_cipher[0])
        for ell in range(1, m):
            j_minus = (j-1) % n_tilde

            sum_01 = ct_part[(j_minus, ell, 0)] + ct_part[(j, ell, 1)]
            sum_23 = ct_part[(j_minus, ell, 2)] + ct_part[(j, ell, 3)]
    
            ctCPrime_j += sum_01
            CtCDPrime_j += sum_23
        
        rotated_dprime = rotate(CtCDPrime_j, -n)
        
        C_cipher[j] = C_cipher[j] + ctCPrime_j + rotated_dprime
    
    return C_cipher


if __name__ == "__main__":
    n = 32
    m_array = powers_of_two_up_to_n(n)

    for m in m_array:
        c_array = powers_of_two_up_to_n(m)
        for c in c_array:
            A = np.arange(1, n * m + 1).reshape(n, m)
            B = np.arange(1, n * m + 1).reshape(m,n)

            # print("=== A ===")
            # print(A)
            # print("=== B ===")
            # print(B)

            # -------------------------------------------------------------
            # 2) A의 lower diagonal 벡터를 c개 이어붙여 A_cipher를 만듦
            #    B도 lower diagonal을 c번 replicate해서 B_cipher를 만듦
            # -------------------------------------------------------------
            A_cipher = []
            A_j_conc = []
            for j in range(n//c):
                for i in range(c):
                    ldv = lower_diagonal_vector_prime(A, j * c + i) 
                    A_j_conc.append(ldv)
                A_cipher.append( np.concatenate(A_j_conc) )
                A_j_conc.clear()

            B_cipher = []
            for ell in range(m):
                ldvB = lower_diagonal_vector(B, ell)
                replicate = np.concatenate([ldvB] * c)
                B_cipher.append(replicate)

            # print("\n=== A_cipher, B_cipher ===")
            # for j in range(n//c):
            #     print(f"[*] A_cipher[j] =", A_cipher[j])
            # print()
            # for ell in range(m):
            #     print(f"[*] B_cipher[{ell}] = {B_cipher[ell]}")

            # -------------------------------------------------------------
            # ciphertext-ciphertext matmul
            # -------------------------------------------------------------
            C_result = cipher_cipher_matmul(A_cipher, B_cipher, n, m, c)
            # print("\n=== C_cipher (암호문 결과) ===")
            # for j in range(len(C_result)):
            #     print(f" - ct.C_{j} = {C_result[j]}  (shape={C_result[j].shape})")

            C_lower = np.zeros((m, n))
            for i in range(m//c):
                for j in range(c):
                    C_lower[i*c+j] = C_result[i][j*n:(j+1)*n]

            # -------------------------------------------------------------
            # 4) 평문 곱셈과 비교 (np.dot)
            # -------------------------------------------------------------
            # print("\n=== np.dot(A,B) (평문 결과) ===")
            C = np.dot(A,B)
            # print(C)
            # print()

            # print("lower diagonal vector of result")

            C_real = np.zeros((m,n))
            for i in range(0, m):
                C_real[i] = lower_diagonal_vector(C, i)

            if (np.array_equal(C_real, C_lower)):
                print(f"Pass at n = {n}, m = {m}, c = {c}")
            else:
                raise ValueError(f"fail at n = {n}, m = {m}, c = {c}")