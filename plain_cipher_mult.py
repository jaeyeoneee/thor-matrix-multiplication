import numpy as np
from func import *
from diagonal_vector import *

def mat_to_lower_diags_head( mat, n, m, c):
    """
    mat: m x n size matrix(n>=m) for attention head input - 각 헤드가 들어와서 slot 형태가 된다.
    """
    # 행렬 shape
    r, s = mat.shape

    # mat을 m x n으로 패딩한다.
    pad_rows = n - s
    pad_cols = m - r
    # print(pad_rows, pad_cols)
    # print(pad_rows, pad_cols)
    mat_padded = np.pad(mat,
                        ((0, pad_cols),   # 아래쪽 행
                        (0, pad_rows)),  # 오른쪽 열
                        mode='constant',
                        constant_values=0)
    
    print("padded matrix shape:", mat_padded.shape )
    
    # diagonal vector extraction
    diags = []
    for i in range(m):
        diags.append(lower_diagonal_vector(mat_padded, i))
    
    # m개의 diagonal을 slot에 c개 만큼 넣어준다.
    slots = []
    num_slots = m // c
    gap = 2**15// (n*c)
    for i in range(num_slots):
        slot = np.zeros((2**15,))
        arr = np.concatenate(diags[i*c: (i+1)*c])
        slot[::gap] = arr
        slots.append(slot)

    return slots


def lower_diag_to_mat_head( C_cipher, n, m, c, origin):
    """
    attention head output to matrix format
    C_cipher: attention haed output
    n, m: padded matrix size (n >= m)
    c: the number of stacked diagonal vectors in each slot
    origin: original matrix size (before padding) to make padded matrix into original size
    """
    
    gap = 2**15 // (n*c)
    num_slots = m // c
    
    diags = [None] * m
    for s_idx in range(num_slots):
        slot = C_cipher[s_idx]
        arr = slot[::gap][:n*c]
        blocks = arr.reshape(c, n)
        for delta in range(c):
            diag_idx = s_idx * c + delta
            diags[diag_idx] = blocks[delta]
    
    mat = np.zeros((m, n), dtype = diags[0].dtype)
    for k in range(m):
        vec = diags[k]
        for t in range(n):
            i = (t+k) % m
            j = t
            mat[i, j] = vec[t]
            
    return mat[:, :origin]


def plaintext_encoding(mat, d, d_h, c=64):
    """
    mat: d_h x d_h size plaintext matrix
    num: d // d_h
    c: the number of diagonals of each slt
    for specific case! needs to be moified for general case.
    """
    
    num = d // d_h
    slot = np.zeros((2**15,))
    slots = []
    gap = 2**15 // (d * c)
    for j in range(d_h):
        slot = np.zeros((2**15,))
        diags = []
        odiag = upper_diagonal_vector(mat, j)
        for i in range(c):
            stacked = []
            diag = np.roll(odiag, -i)
            for _ in range(num):
                stacked.append(diag)
            concat = np.concatenate(stacked)
            diags.append(concat)
        slot[::gap] = np.concatenate(diags)
        slots.append(slot)
    # print("slot[0]", slots[0][0*gap], slots[0][(256*2-1)*gap], slots[0][(256*3-2)*gap])
    print("slot shape:", len(slots))
    return slots
            

def plain_cipher_mult(A_diags, B_diags, d, n, d_h, c=64):
    
    gap = 2**15 // (d * c)
    rs = []
    print("A_diags shape:", len(A_diags), len(A_diags[0]), len(A_diags[0][0]))
    for j in range(len(A_diags)):
        slot = np.zeros((2**15, ))
    
        A_diag = A_diags[j]
        for i in range(n//d_h):
            block_diag = A_diag[i]
            # print(len(block_diag))
            B_diag = B_diags[i]
            for k in range(d_h):
                slot += block_diag[k] * np.roll(B_diag, -d * k * gap)
        rs.append(slot)
        
    return rs
        
    

if __name__ == "__main__":
    """
    d는 n으로 나누어진다.
    A의 크기는 dxd이고, B의 크기는 nxd이다. 
    """
    d = 256
    d_h = 64
    n = 192
    c = 64
    num = n // d_h
    
    # A, B 행렬 생성
    A = np.arange(1, n*n+1).reshape(n, n)
    B = np.arange(1, n * d + 1).reshape(n, d)
    
    # B를 head별 diagonal vector로 변환
    diags_B = []
    for i in range(num):
        diag = mat_to_lower_diags_head(B[d_h*i:d_h*(i+1), :], d, d_h, c)
        diags_B.append(diag[0])
        print(diag[0])
    print("number of the diagonals:", len(diags_B))
    print("--------matrix B --------------------")
    print(B)
    print("--------diagonal vectors of B --------")
    print(diags_B)
    
    
    # A를 block 별 diagonal 생성
    print("---------matrix A --------------------")
    plaintext_encoding(A[:d_h, :d_h], d, d_h, c)
    print("--------matrix A[:d_h, :d_h] --------------------")
    print(A[:d_h, :d_h])
    print("--------diagonal vectors of A --------")
    diags_A = plaintext_encoding(A[:d_h, :d_h], d, d_h, c)
    print(diags_A)
    
    diags_A = []
    for i in range(num):
        diags = []
        for j in range(num):
            diag = plaintext_encoding(A[d_h*i:d_h*(i+1), d_h*j:d_h*(j+1)], d, d_h, c)
            diags.append(diag)
        diags_A.append(diags)
    
    print("-------------matrix multiplicaion result----------")
    print("real multiplication result")
    print(np.matmul(A, B)[:64, :])
    print("diagonal multiplication result")
    rs = plain_cipher_mult(diags_A, diags_B, d, n, d_h, c)
    print(rs)
    print(len(rs))
    
    print("-------------diagonal result comparison ----------")
    mat = np.matmul(A, B)[:64, :]
    print("diagonal 0")
    print("real result:", lower_diagonal_vector(mat, 1))
    print("diagonal multiplication result:", rs[0][::2][256:256*2])
    for i in range(1, num):
        mat = np.matmul(A, B)[d_h*i:d_h*(i+1), :]
        for j in range(1, d_h):
            assert np.allclose(lower_diagonal_vector(mat, j), rs[i][::2][256*j:256*(j+1)])
    
    print("-------------diagonal result to matrix ----------")
    full_rows = []
    
    for i in range(num):
        full_rows.append(lower_diag_to_mat_head([rs[i]], d, d_h, c, d))
    
    combined_mat = np.vstack(full_rows)
    
    assert np.allclose(combined_mat, np.matmul(A, B))
    
    