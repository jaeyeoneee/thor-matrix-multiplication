"""
masking 결과 확인.
"""

import numpy as np

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

n = 4
c = 2
for l in range(n):
    mu0, mu1, mu2, mu3 = mask(n, c, l)
    print("l: ", l)
    print("mu0:", mu0)
    print("mu1:", mu1)
    print("mu2:", mu2)
    print("mu3:", mu3)