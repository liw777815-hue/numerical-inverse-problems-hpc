from numpy import zeros, array, empty, dot, arange, int32, float64
from mpi4py import MPI

comm = MPI.COMM_WORLD
numprocs = comm.Get_size()
rank = comm.Get_rank()

if rank == 0 :
    # Считываем из файла число строк M и число столбцов N
    f1 = open('in.dat', 'r')
    N = array(int32(f1.readline()))
    M = array(int32(f1.readline()))
    f1.close()
else :
    N = array(1, dtype=int32)
    M = array(1, dtype=int32)

if rank == 0 :
    counts_M = empty(numprocs, dtype=int32)
    displs_M = empty(numprocs, dtype=int32)
    ave, res = divmod(M, numprocs)
    displs_M[0] = 0
    for k in range(numprocs) :
        if k < res :
            counts_M[k] = ave + 1
        else :
            counts_M[k] = ave
        if k >= 1 :
            displs_M[k] = displs_M[k-1] + counts_M[k-1]
else :
    counts_M = None
    displs_M = None
    
if rank == 0 :
    counts_N = empty(numprocs, dtype=int32)
    displs_N = empty(numprocs, dtype=int32)
    ave, res = divmod(N, numprocs)
    displs_N[0] = 0
    for k in range(numprocs) :
        if k < res :
            counts_N[k] = ave + 1
        else :
            counts_N[k] = ave
        if k >= 1 :
            displs_N[k] = displs_N[k-1] + counts_N[k-1]    
else :
    counts_N = empty(numprocs, dtype=int32)
    displs_N = None
    
comm.Bcast(counts_N, root=0)

M_part = array(1, dtype=int32)
comm.Scatter(counts_M, M_part, root=0)

N_part = array(1, dtype=int32)
comm.Scatter(counts_N, N_part, root=0)

comm.Bcast(N, root=0)

# Считываем из файла матрицу A
if rank == 0 :
	f2 = open('AData.dat', 'r')
	A_part = empty((M_part,N))
	for j in range(M_part) :
		for i in range(N) :
			A_part[j,i] = float(f2.readline())
	for k in range(1, numprocs) :
		A_part_temp = empty((counts_M[k],N))
		for j in range(counts_M[k]) :
			for i in range(N) :
				A_part_temp[j,i] = float(f2.readline())
		comm.Send([A_part_temp, counts_M[k]*N, MPI.DOUBLE], dest=k)
		del A_part_temp
	f2.close()
else :
	A_part = empty((M_part,N))
	comm.Recv([A_part, M_part*N, MPI.DOUBLE], source=0)

# Считываем из файла вектор b
if rank == 0 : 
    b = empty(M, dtype=float64)
    f3 = open('bData.dat', 'r')
    for j in range(M) :
        b[j] = float(f3.readline())
    f3.close()
else :
    b = None

def сonjugate_gradient_method(A_part, b, x, N,
                              N_part, M_part, counts_M, displs_M,
                              counts_N, displs_N, rank, comm) :
    s = 1
    # print(f'It is OK', flush=True)
    # p_part = zeros(N_part, dtype=float64)
    while s <= N :
        if s == 1 :
            comm.Bcast([x, N, MPI.DOUBLE], root=0)
            p_part = zeros(N_part, dtype=float64)
            # r = dot(A.T, dot(A,x) - b)
            c_part = dot(A_part, x)
            if rank == 0 :
                c = empty(M, dtype=float64)
            else :
                c = None
            comm.Gatherv([c_part, M_part, MPI.DOUBLE],
                         [c, counts_M, displs_M, MPI.DOUBLE], root=0)
            print(f'It is OK', flush=True)
            if rank == 0 :
                c = c - b
            comm.Scatterv([c, counts_M, displs_M, MPI.DOUBLE], 
                          [c_part, M_part, MPI.DOUBLE], root=0)
            r_temp = dot(A_part.T, c_part)
            r_part = empty(N_part, dtype=float64)
            comm.Reduce_scatter([r_temp, N, MPI.DOUBLE],
                                [r_part, N_part, MPI.DOUBLE],
                                recvcounts=counts_N , op=MPI.SUM)
            # print(f'It is OK', flush=True)
        else :
            r_part = r_part - q_part / ScalP
            # r = r - q/dot(p, q)
        # p = p + r/dot(r, r)
        ScalP_temp = array(dot(r_part, r_part), dtype=float64)
        ScalP = array(0, dtype=float64)
        comm.Allreduce([ScalP_temp, 1, MPI.DOUBLE], 
                       [ScalP, 1, MPI.DOUBLE], op=MPI.SUM)
        p_part = p_part + r_part / ScalP
        # q = dot(A.T, dot(A, p))
        p = empty(N, dtype=float64)
        comm.Allgatherv([p_part, N_part, MPI.DOUBLE], 
                        [p, counts_N, displs_N , MPI.DOUBLE])
        d_part = dot(A_part, p)
        q_temp = dot(A_part.T, d_part)
        q_part = empty(N_part, dtype=float64)
        comm.Reduce_scatter([q_temp, N, MPI.DOUBLE],
                            [q_part, N_part, MPI.DOUBLE],
                            recvcounts=counts_N , op=MPI.SUM)
        # x = x - p/dot(p,q)
        ScalP_temp = array(dot(p_part, q_part), dtype=float64)
        comm.Allreduce([ScalP_temp, 1, MPI.DOUBLE],
                       [ScalP, 1, MPI.DOUBLE], op=MPI.SUM)
        if rank == 0:
            x = x - p/ScalP

        s = s + 1
    return x

if rank == 0 :
    x = zeros(N, dtype=float64)
else :
    x = empty(N, dtype=float64)

x = сonjugate_gradient_method(A_part, b, x, N,
                              N_part, M_part, counts_M, displs_M,
                              counts_N, displs_N, rank, comm)

if rank == 0 :
    from matplotlib.pyplot import figure, axes, show
    fig = figure()
    ax = axes(xlim=(0, N), ylim=(-1.5, 1.5))
    ax.set_xlabel('i'); ax.set_ylabel('x[i]')
    ax.plot(arange(N), x, '-k', lw=2)
    show()
