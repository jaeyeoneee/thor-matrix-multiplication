import numpy as np
from func import *
from diagonal_vector import *


def build_subblocks_diagonal_order(A, n):
    """
    (d x d) 행렬 A를 (n x n) 크기로 나누고,
    i = 0..(d_blocks-1)에 대해,
      A^(i) = { A_{i,0}, A_{i+1,1}, ..., A_{i + d_blocks-1, d_blocks-1} }
    형태로 묶어서 리턴한다.

    Return 형태 예:
    diag_subblocks[i] = [ A_{i,0}, A_{i+1,1}, ..., A_{i + (d_blocks-1), (d_blocks-1)} ]
    """
    d = A.shape[0]
    d_blocks = d // n
    diag_subblocks = []

    for i in range(d_blocks):
        sublist = []
        for offset in range(d_blocks):
            p = i + offset
            q = offset
            # 인덱스가 d_blocks 넘어가면 mod 연산할 수도 있고,
            # 보통은 i+offset < d_blocks 범위 안에서만 쓰기도 한다.
            # 여기서는 mod로 처리
            p_mod = p % d_blocks
            q_mod = q % d_blocks

            block = get_submatrix(A, n, p_mod, q_mod)
            sublist.append(block)
        diag_subblocks.append(sublist)
    return diag_subblocks

def pack_upper_diags_in_interlaced_form(subblock_list, n, c):
    """
    subblock_list: A^(i)에 해당하는 n x n sub-block들의 리스트 예:
      [ A_{i,0}, A_{i+1,1}, ..., A_{...,...} ]
      각각 (n x n) 크기
    n: 각 sub-block의 행/열 크기
    c: 한 ciphertext에 몇 개 diagonal을 묶을지
    name: 디버깅용 태그("A" 등)

    여기서는
    pt.A_{i, j, l, r}
    꼴로 저장한다고 가정. 실제론 4차원 배열이거나 dict 등을 써도 됨.

    간단히, 각 sub-block에서 k=0..(n-1) upper diag를 뽑아서,
    [c]개씩 interlace/pack하는 식만 스케치.
    (실제로는 "j, r" 인덱스 등도 필요하지만, 여기서는 형태만 보여줌.)
    """
    result = dict()

    # i는 총 block의 개수
    for i in range(d//n):
        subblock = subblock_list[i]
        # j는 subblock 안에서
        for j in range(n//c):
            for l in range(c):
                k = l + c * j
                upper_diag = np.zeros(d)
                for tmp in range(d):
                    upper_diag[tmp] = upper_diagonal_vector(subblock[tmp % (d//n)], k)[tmp // (d//n)]
                # print(upper_diag) 
                rotated_upper_diag = np.zeros(d * c)
                for r in range(n//c):
                    for r_prime in range(c):
                        rotated_upper_diag[(r_prime)*d: (r_prime+1)*d] = rotate(upper_diag, r_prime*c)
                    print(rotated_upper_diag)
                # print(rotated_upper_diag)
                    
                    




if __name__ == "__main__":
    """
    d는 n으로 나누어진다.
    A의 크기는 dxd이고, B의 크기는 nxd이다. 
    """
    d = 8
    n = 4
    c = 2
    A = np.arange(1, d * d + 1).reshape(d, d)
    d_blocks = d // n

    print("=== A ===")
    print(A)

    diag_subblocks = build_subblocks_diagonal_order(A, n)
    # print("\n=== A^(i) ===")
    # for i in range(d_blocks):
    #     print(f"[*] A^{i} = {diag_subblocks[i]}")
    
    pack_upper_diags_in_interlaced_form(diag_subblocks, n, c)
