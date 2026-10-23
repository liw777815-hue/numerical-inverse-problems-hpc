
import numpy as np
from mpi4py import MPI

comm = MPI.COMM_WORLD
numprocs = comm.Get_size()
num_row = num_col = int(np.sqrt(numprocs))

dims = (num_row, num_col)
comm_cart = comm.Create_cart(dims, periods=(True, True), reorder=True)
rank_cart = comm_cart.Get_rank()
my_row, my_col = comm_cart.Get_coords(rank_cart)
comm_col = comm_cart.Split(rank_cart % num_col, rank_cart)
comm_row = comm_cart.Split(rank_cart // num_row, rank_cart)


def auxiliary_arrays(M, num):
    ave, res = divmod(M, num)
    rcounts = np.empty(num, dtype=np.int32)
    displs = np.empty(num, dtype=np.int64)
    for k in range(num):
        if k < res:
            rcounts[k] = ave + 1
        else:
            rcounts[k] = ave
        if k == 0:
            displs[k] = 0
        else:
            displs[k] = displs[k - 1] + rcounts[k - 1]
    return rcounts, displs


M = N = 10
rcounts_M, displs_M = auxiliary_arrays(M, num_row)
rcounts_N, displs_N = auxiliary_arrays(N, num_col)
M_part = int(rcounts_M[my_row])
N_part = int(rcounts_N[my_col])
rows = np.arange(displs_M[my_row], displs_M[my_row] + M_part)
cols = np.arange(displs_N[my_col], displs_N[my_col] + N_part)


A_part = np.array([np.random.default_rng(42 + int(i)).integers(-5, 6, N)[cols]for i in rows], dtype=float)

B = np.empty((M_part, 0))
C = np.empty((N_part, 0))
I = np.ones(M, dtype=bool)
J = np.ones(N, dtype=bool)
eps = 1e-10
rng = np.random.default_rng(123)


def owner(k, displs):
    for p in range(len(displs)):
        if k >= displs[p] and (p == len(displs) - 1 or k < displs[p + 1]):
            return p

pair_dtype = np.dtype([("value", np.float64), ("index", np.int32)], align=True)

def maxloc(values, indices):
    local_max = np.array([-np.inf], dtype=np.float64)
    local_index = np.array([np.iinfo(np.int32).max], dtype=np.int32)
    if len(indices) > 0:
        k = np.argmax(np.abs(values))
        local_max[0] = abs(values[k])
        local_index[0] = indices[k]
    global_max = np.empty_like(local_max)
    comm_cart.Allreduce(local_max, global_max, op=MPI.MAX)
    if local_max[0] != global_max[0]:
        local_index[0] = np.iinfo(np.int32).max
    global_index = np.empty_like(local_index)
    comm_cart.Allreduce(local_index, global_index, op=MPI.MIN)
    return int(global_index[0])


def residual_column(j):
    return A_part[:, j - displs_N[my_col]] - B @ C[j - displs_N[my_col], :]


def residual_row(i):
    return A_part[i - displs_M[my_row], :] - C @ B[i - displs_M[my_row], :]


def bc_norm():
    r = B.shape[1]
    local = np.zeros((2, r, r), dtype=np.float64)
    if my_col == 0:
        local[0] = B.T @ B
    if my_row == 0:
        local[1] = C.T @ C
    gram = np.empty_like(local)
    comm_cart.Allreduce(local, gram, op=MPI.SUM)
    return np.sqrt(max(0.0, np.sum(gram[0] * gram[1])))

if rank_cart == 0:
    A_full = np.array([np.random.default_rng(42 + i).integers(-5, 6, N)for i in range(M)])
    print(" A：\n", A_full)

r = 0
while r < min(M, N):
    j_buf = np.zeros(1, dtype=np.int32)
    if rank_cart == 0:
        j_buf[0] = rng.choice(np.flatnonzero(J))
    comm_cart.Bcast(j_buf, root=0)
    j = int(j_buf[0])
    for _ in range(2):
        if my_col == owner(j, displs_N):
            mask = I[rows]
            i = maxloc(residual_column(j)[mask], rows[mask])
        else:
            i = maxloc(np.empty(0), np.empty(0, dtype=int))

        if my_row == owner(i, displs_M):
            mask = J[cols]
            j = maxloc(residual_row(i)[mask], cols[mask])
        else:
            j = maxloc(np.empty(0), np.empty(0, dtype=int))

    pivot = np.zeros(1, dtype=np.float64)
    if my_row == owner(i, displs_M) and my_col == owner(j, displs_N):
        pivot[0] = residual_row(i)[j - displs_N[my_col]]
    comm_cart.Bcast(pivot, root=comm_cart.Get_cart_rank((owner(i, displs_M), owner(j, displs_N))))
    norm_BC = bc_norm()
    estimate = abs(pivot[0]) * np.sqrt((M - r - 1) * (N - r - 1))
    if estimate <= eps * norm_BC:
        if rank_cart == 0:
            print(f"停止：r={r}")
        break
    b = np.empty(M_part, dtype=np.float64)
    if my_col == owner(j, displs_N):
        b[:] = residual_column(j)
    comm_row.Bcast(b, root=owner(j, displs_N))

    c = np.empty(N_part, dtype=np.float64)
    if my_row == owner(i, displs_M):
        c[:] = residual_row(i) / pivot[0]
    comm_col.Bcast(c, root=owner(i, displs_M))
    B = np.column_stack((B, b))
    C = np.column_stack((C, c))
    I[i] = False
    J[j] = False
    r += 1

if rank_cart == 0:
    print(f"r={r}, 主元(i,j)=({i},{j}), estimate={estimate:.3e}")
