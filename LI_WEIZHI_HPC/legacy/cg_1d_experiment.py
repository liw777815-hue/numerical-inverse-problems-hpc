from mpi4py import MPI
import numpy as np

comm = MPI.COMM_WORLD
numprocs = comm.Get_size()
rank = comm.Get_rank()


def conjugate_gradient_method(A_part, b_part, x_part, N_part, rcounts_N, displs_N, comm, N):
    x = np.empty(N, dtype=np.float64)
    p = np.empty(N, dtype=np.float64)
    r_part = np.empty(N_part, dtype=np.float64)
    p_part = np.empty(N_part, dtype=np.float64)
    q_part = np.empty(N_part, dtype=np.float64)

    ScalP = np.array(0, dtype=np.float64)
    ScalP_temp = np.empty(1, dtype=np.float64)

    s = 1
    p_part[:] = np.zeros(N_part, dtype=np.float64)

    while s <= N:
        if s == 1:
            comm.Allgatherv([x_part, N_part, MPI.DOUBLE],
                            [x, rcounts_N, displs_N, MPI.DOUBLE])
            Ax_local = np.dot(A_part, x) 
            r_temp = np.dot(A_part.T, Ax_local - b_part)
            comm.Reduce_scatter([r_temp, MPI.DOUBLE],
                                [r_part, MPI.DOUBLE],
                                op=MPI.SUM)
        else:
            ScalP_temp[0] = np.dot(p_part, q_part)
            comm.Allreduce([ScalP_temp, MPI.DOUBLE],
                           [ScalP, MPI.DOUBLE], op=MPI.SUM)

            r_part = r_part - q_part / ScalP

            ScalP_temp[0] = np.dot(r_part, r_part)
            comm.Allreduce([ScalP_temp, MPI.DOUBLE],
                           [ScalP, MPI.DOUBLE], op=MPI.SUM)
            p_part = p_part + r_part / ScalP
        comm.Allgatherv([p_part, N_part, MPI.DOUBLE],
                        [p, rcounts_N, displs_N, MPI.DOUBLE])
        Ap_local = np.dot(A_part, p)
        q_temp = np.dot(A_part.T, Ap_local)
        comm.Reduce_scatter([q_temp, MPI.DOUBLE],
                            [q_part, MPI.DOUBLE],
                            op=MPI.SUM)
        ScalP_temp[0] = np.dot(p_part, q_part)
        comm.Allreduce([ScalP_temp, MPI.DOUBLE],
                       [ScalP, MPI.DOUBLE], op=MPI.SUM)
        x_part = x_part - p_part / ScalP
        s = s + 1

    return x_part

def auxiliary_arrays(total_size, numprocs):
    ave, res = divmod(total_size, numprocs)
    rcounts = np.empty(numprocs, dtype=np.int32)
    displs = np.empty(numprocs, dtype=np.int32)

    displs[0] = 0
    for k in range(numprocs):
        if k < res:
            rcounts[k] = ave + 1
        else:
            rcounts[k] = ave
        if k > 0:
            displs[k] = displs[k - 1] + rcounts[k - 1]
    return rcounts, displs

if __name__ == "__main__":
    if rank == 0:
        print("Reading problem parameters...")
        with open('in.dat', 'r') as f:
            N = int(f.readline())
            M = int(f.readline())
        print(f"N={N}, M={M}")
    else:
        N, M = 0, 0
    N = comm.bcast(N, root=0)
    M = comm.bcast(M, root=0)
    rcounts_M, displs_M = auxiliary_arrays(M, numprocs)
    rcounts_N, displs_N = auxiliary_arrays(N, numprocs)
    M_part = rcounts_M[rank]
    N_part = rcounts_N[rank]
    if rank == 0:
        print("Reading matrix A...")
        A = np.empty((M, N), dtype=np.float64)
        with open('AData.dat', 'r') as f:
            for j in range(M):
                for i in range(N):
                    A[j, i] = float(f.readline())
    else:
        A = None
    A_part = np.empty((M_part, N), dtype=np.float64)
    comm.Scatterv([A, rcounts_M, displs_M, MPI.DOUBLE],
                  [A_part, MPI.DOUBLE], root=0)
    if rank == 0:
        print("Reading vector b...")
        b = np.empty(M, dtype=np.float64)
        with open('bData.dat', 'r') as f:
            for j in range(M):
                b[j] = float(f.readline())
    else:
        b = None
    b_part = np.empty(M_part, dtype=np.float64)
    comm.Scatterv([b, rcounts_M, displs_M, MPI.DOUBLE],
                  [b_part, MPI.DOUBLE], root=0)
    if rank == 0:
        x = np.zeros(N, dtype=np.float64)
    else:
        x = None

    x_part = np.empty(N_part, dtype=np.float64)
    comm.Scatterv([x, rcounts_N, displs_N, MPI.DOUBLE],
                  [x_part, MPI.DOUBLE], root=0)
    if rank == 0:
        print("Starting conjugate gradient method...")

    x_part = conjugate_gradient_method(A_part, b_part, x_part, N_part,
                                       rcounts_N, displs_N, comm, N)
    if rank == 0:
        x_final = np.empty(N, dtype=np.float64)
    else:
        x_final = None

    comm.Gatherv([x_part, N_part, MPI.DOUBLE],
                 [x_final, rcounts_N, displs_N, MPI.DOUBLE], root=0)

    if rank == 0:
        print("Solution found!")
        print(f"First 10 elements of solution: {x_final[:10]}")
        try:
            import matplotlib.pyplot as plt

            plt.figure(figsize=(10, 6))
            plt.plot(range(N), x_final, 'b-', linewidth=2)
            plt.xlabel('Index i')
            plt.ylabel('Solution x[i]')
            plt.title('Solution of Linear System using Conjugate Gradient Method')
            plt.grid(True)
            plt.savefig('solution.png')
            print("Solution plot saved as solution.png")
        except ImportError:
            print("Matplotlib not available, skipping plot generation")
