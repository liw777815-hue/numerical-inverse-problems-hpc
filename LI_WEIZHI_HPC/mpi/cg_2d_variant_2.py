from mpi4py import MPI
import numpy as np
from numpy import linspace
import matplotlib.pyplot as plt
from numpy.matlib import empty

comm = MPI.COMM_WORLD
numprocs = comm.Get_size()
num_row = num_col = np.int32(np.sqrt(numprocs))
dims=(num_row, num_col)
comm_cart=comm.Create_cart(dims,periods=(True,True),reorder=True)
rank_cart=comm_cart.Get_rank()
my_row,my_col = comm_cart.Get_coords(rank_cart)
comm_col = comm_cart.Split(rank_cart % num_col,rank_cart)
comm_row = comm_cart.Split(rank_cart // num_row,rank_cart)
rank_col=comm_col.Get_rank()
rank_row=comm_row.Get_rank()


def conjugate_gradient_method(A_part, b_part, x_part, N_part, M_part, comm_row, comm_col, N, delta_true):


    z_part = np.empty(N_part, dtype=np.float64)
    p_part = np.empty(N_part, dtype=np.float64)

    scal_temp = np.zeros(1, dtype=np.float64)
    scal_temp_1 = np.zeros(1, dtype=np.float64)
    scal_result = np.zeros(1, dtype=np.float64)
    scal_result_1 = np.zeros(1, dtype=np.float64)
    b_part_org=b_part.copy()

    Ax_temp = np.dot(A_part, x_part)
    Ax_part = np.empty(M_part, dtype=np.float64)
    comm_row.Allreduce(Ax_temp, Ax_part, op=MPI.SUM)
    r_part = b_part - Ax_part

    z_temp = np.dot(A_part.T, r_part)
    comm_col.Allreduce(z_temp, z_part, op=MPI.SUM)


    p_part = z_part.copy()


    r_part_end=np.empty(M_part, dtype=np.float64)
    r_part_row_end=np.empty(M_part, dtype=np.float64)
    request_1=comm_row.Allreduce_init(r_part_end, r_part_row_end, op=MPI.SUM)

    r_part_row_value=empty(1,dtype=np.float64)
    r_global_end = np.empty(1, dtype=np.float64)
    request_2=comm_col.Allreduce_init(r_part_row_value, r_global_end, op=MPI.SUM)

    request_3=comm_row.Allreduce_init(scal_temp, scal_result, op=MPI.SUM)

    Ap_temp=np.empty(M_part, dtype=np.float64)
    Ap_part = np.empty(M_part, dtype=np.float64)
    request_4=comm_row.Allreduce_init(Ap_temp, Ap_part, op=MPI.SUM)

    request_5=comm_col.Allreduce_init(scal_temp_1, scal_result_1, op=MPI.SUM)
    request_6=comm_row.Allreduce_init(scal_temp, scal_result, op=MPI.SUM)

    z_new = np.empty(N_part, dtype=np.float64)
    z_new_temp=np.empty(N_part,dtype=np.float64)
    request_7=comm_col.Allreduce_init(z_new_temp, z_new, op=MPI.SUM)


    s = 1
    while s <= N:
        r_part_end [:]= np.dot(A_part, x_part)
        # r_part_row_end = np.empty(M_part, dtype=np.float64)
        # comm_row.Allreduce(r_part_end, r_part_row_end, op=MPI.SUM)
        MPI.Prequest.Start(request_1)
        MPI.Prequest.Wait (request_1)

        r_part_row_value[:] = np.linalg.norm(r_part_row_end - b_part_org) ** 2
        # r_global_end = np.empty(1, dtype=np.float64)
        # comm_col.Allreduce(r_part_row_value, r_global_end, op=MPI.SUM)
        MPI.Prequest.Start(request_2)
        MPI.Prequest.Wait(request_2)

        if r_global_end < delta_true ** 2:
            break

        scal_temp[0] = np.dot(z_part, z_part)
        # comm_row.Allreduce(scal_temp, scal_result, op=MPI.SUM)
        MPI.Prequest.Start(request_3)


        Ap_temp[:] = np.dot(A_part, p_part)
        # comm_row.Allreduce(Ap_temp, Ap_part, op=MPI.SUM)
        MPI.Prequest.Start(request_4)
        MPI.Request.Wait(request_4)


        scal_temp_1[0] = np.dot(Ap_part, Ap_part)
        # comm_col.Allreduce(scal_temp, scal_result, op=MPI.SUM)
        MPI.Prequest.Start(request_5)
        MPI.Request.Wait(request_5)
        Ap_norm_sq = scal_result_1[0]
        MPI.Request.Wait(request_3)
        zz_old = scal_result[0]


        k1 = zz_old / Ap_norm_sq

        x_part += k1 * p_part
        r_part -= k1 * Ap_part

        z_new_temp[:] = np.dot(A_part.T, r_part)
        # z_new = np.empty(N_part, dtype=np.float64)
        # comm_col.Allreduce(z_new_temp, z_new, op=MPI.SUM)
        MPI.Prequest.Start(request_7)
        MPI.Request.Wait(request_7)
        scal_temp[0] = np.dot(z_new, z_new)


        # comm_row.Allreduce(scal_temp, scal_result, op=MPI.SUM)
        MPI.Prequest.Start(request_6)
        MPI.Request.Wait(request_6)
        zz_new = scal_result[0]



        k2 = zz_new / zz_old

        p_part = z_new + k2 * p_part
        z_part = z_new

        s += 1
    return x_part, s

def auxiliary_arrays(M,num):
    ave,res =divmod(M,num)
    rcounts= np.empty(num,dtype=np.int32)
    displs = np.empty(num,dtype=np.int64)
    for k in range(num):
        if k < res:
            rcounts[k] = ave +1
        else :
            rcounts[k] = ave
        if k == 0 :
            displs[k] = 0
        else :
            displs[k] = displs[k-1] + rcounts[k-1]
    return rcounts,displs



def rho_model(s):
    return 2 * np.exp(-(s - 0.382) ** 2 / 0.009) + \
        1.2 * np.exp(-(s - 0.618) ** 2 / 0.018)



def system_matrix(s,N_part, h, x, M_part,displs_M,displs_N,l_y, l_z,my_row,my_col):
    A_part = np.zeros((M_part, N_part), dtype=np.float64)  #
    G = -6.67e-11
    start_row_M = displs_M[my_row]
    end_row_M =start_row_M + M_part-1
    start_col_N = displs_N[my_col]
    end_col_N =start_col_N + N_part-1
    if my_col== 0 :
        for global_i in range(start_row_M, end_row_M + 1):
            local_i = global_i - start_row_M
            A_part[local_i, 0] = -1/2 * G * h / ((x[global_i] - s[start_col_N]) ** 2 + l_y ** 2 + l_z ** 2) ** 0.5
            for j in range(1, N_part):
                A_part[local_i, j] = -G * h / ((x[global_i] - s[start_col_N + j]) ** 2 + l_y ** 2 + l_z ** 2) ** 0.5
    if my_col == num_col - 1:
        for global_i in range(start_row_M, end_row_M + 1):
            local_i = global_i - start_row_M
            for j in range(0, N_part - 1):
                A_part[local_i, j] = -G * h / ((x[global_i] - s[start_col_N + j]) ** 2 + l_y ** 2 + l_z ** 2) ** 0.5
            A_part[local_i, N_part - 1] = -1 / 2 * G * h / (
                        (x[global_i] - s[end_col_N]) ** 2 + l_y ** 2 + l_z ** 2) ** 0.5

    else :
        for global_i in range(start_row_M, end_row_M + 1):
            local_i = global_i - start_row_M
            for j in range(0, N_part):
                A_part[local_i, j] = -G * h / ((x[global_i] - s[start_col_N + j]) ** 2 + l_y ** 2 + l_z ** 2) ** 0.5

    return A_part



M = 200
N = 50
if rank_cart == 0:
    a, b = 0., 1.
    N_model = 100
    h_model = (b - a) / N_model
    s_model = linspace(a, b, N_model + 1)
    X_model = np.array([rho_model(si) for si in s_model])
    c, d = -1., 2.

    x_grid = linspace(c, d, M + 1)
    l_y, l_z = 0.2, 0.1
    x = linspace(c, d, M + 1)
    displs0 = np.array([0],dtype=np.int32)

    A_full = system_matrix(s_model, N_model+1, h_model, x, M + 1, displs0,displs0, l_y, l_z,my_row,my_col)
    B = A_full @ X_model
    delta = 0.001
    np.random.seed(1)
    B_delta = B + delta * B * (2 * np.random.random(len(B)) - 1)
    delta_true = np.linalg.norm(B_delta - B)


    h = (b - a) / N
    s = linspace(a, b, N + 1)

    A = system_matrix(s, N+1, h, x, M + 1, displs0,displs0, l_y, l_z,my_row,my_col)
    alpha = 4.1 * 10 ** (-26)
else:
     h = s = x_grid = l_y = l_z = delta_true = B_delta = None
     x=np.empty(M+1)
h = comm_cart.bcast(h, root=0)
s = comm_cart.bcast(s, root=0)
x_grid = comm_cart.bcast(x_grid, root=0)
l_y = comm_cart.bcast(l_y, root=0)
l_z = comm_cart.bcast(l_z, root=0)
delta_true = comm_cart.bcast(delta_true, root=0)



rcounts_M, displs_M = auxiliary_arrays(M+1,num_row)
rcounts_N, displs_N = auxiliary_arrays(N+1,num_col)
M_part = rcounts_M[my_row]
N_part = rcounts_N[my_col]
x=comm_cart.bcast(x, root=0)

A_part = system_matrix(s, N_part, h, x, M_part,displs_M,displs_N,l_y, l_z,my_row,my_col)


B_part =np.empty(M_part, dtype=np.float64)
if my_col == 0:
    comm_col.Scatterv([B_delta,rcounts_M,displs_M,MPI.DOUBLE],[B_part,M_part,MPI.DOUBLE],root=0)
comm_row.Bcast(B_part,root=0)
#
x_0 = np.zeros(N_part, dtype=np.float64)

x_cg_part,iter =conjugate_gradient_method(A_part, B_part, x_0,  N_part, M_part, comm_row, comm_col, N,delta_true)

if rank_cart == 0:
    x_cg=np.empty(N+1, dtype=np.float64)
else :
    x_cg=None
if my_row == 0 :
    comm_row.Gatherv(x_cg_part,[x_cg,rcounts_N,displs_N,MPI.DOUBLE],root=0)
if rank_cart == 0:
    print(iter)

    alpha=4.1*10**(-26)
    x_tikhonov = np.linalg.solve(np.dot(A.T, A) + alpha*np.eye(N+1), np.dot(A.T, B_delta))
    plt.figure()
    plt.plot(s, x_cg, label="CG", lw=2)
    plt.plot(s, x_tikhonov, label="x_tikhonov", lw=2, linestyle='-')
    plt.plot(s_model,X_model,label='x_true', lw=2)
    plt.legend()
    plt.xlabel(" s")
    plt.ylabel(" rho")
    plt.show()
