import numpy as np
from func import *
from diagonal_vector import *

slot = 2**15

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


def _mask_k(n, c):
  
  gap = slot//(n*c)
  
  masks = []
  for k in range(c//2):
    mask = np.zeros((slot,))
    start = 2*k*n*gap
    end = 2*(k+1)*n*gap
    mask[start:end:gap] = 1
    masks.append(mask)
  return masks


def _mask_l(n, c):
  
  gap = slot//(n*c)
  
  mask = np.zeros((slot,))
  for k in range(c//2):
    s = 2*k
    start_idx = s*n * gap
    end_idx = (s+1)*n * gap
    mask[start_idx:end_idx:gap] = 1
  
  return mask
  

def merge_copy(diags, n, m, c):
  
  gap = slot//(n * c)
  ct_B_prime = [np.zeros(slot) for _ in range(m)]
  
  mask_k = _mask_k(n, c)
  mask_l = _mask_l(n, c)
  
  for j in range(m//c):
    for k in range(c//2):
      idx_even = c*j+2*k
      idx_odd = c*j+2*k+1
      
      ctj_k = np.multiply(diags[j], mask_k[k])
      
      for r in range(int(np.log2(c//2))):
        shift = 2 * n * (2**r) * gap
        ctj_k = np.add(ctj_k, np.roll(ctj_k, shift))
        
      ct_B_prime[idx_even] = np.multiply(ctj_k, mask_l)
      ct_B_prime[idx_odd] = np.subtract(ctj_k, ct_B_prime[idx_even])
      
    for k in range(c):
      shift = n*gap
      ct_B_prime[c*j+k] = ct_B_prime[c*j+k] + np.roll(ct_B_prime[c*j+k], shift)
  
  return ct_B_prime


if __name__ == "__main__":
  
  print("-----------mask k test -------")
  m=64
  n=256
  c = 64
  mask = _mask_k(n, c)
  print(mask[0][256*3:256*4+2])
  
  print("-----------test 1: mxn-------------")
  m = 64
  n = 256
  
  A = np.arange(1, m*n+1).reshape(m, n)
  
  diags = mat_to_lower_diags_head(A, n, m, 64)
  
  rs = merge_copy(diags, n, m, c)
  
  print(rs)
  
  # print("----------test 2: nxn--------------")
  # n=256
  
  # B = np.arange(1, n*n+1).reshape(n, n)
  


