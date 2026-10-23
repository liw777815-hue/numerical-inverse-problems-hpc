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
            # 收集所有进程的x_part到完整的x向量
            comm.Allgatherv([x_part, N_part, MPI.DOUBLE],
                            [x, rcounts_N, displs_N, MPI.DOUBLE])

            # 计算残差 r = A^T(Ax - b)
            Ax_local = np.dot(A_part, x)  # 局部矩阵向量乘法
            r_temp = np.dot(A_part.T, Ax_local - b_part)  # 局部转置矩阵向量乘法

            # 归约并分散结果到各进程
            comm.Reduce_scatter([r_temp, MPI.DOUBLE],
                                [r_part, MPI.DOUBLE],
                                op=MPI.SUM)
        else:
            # 计算标量积 (p, q)
            ScalP_temp[0] = np.dot(p_part, q_part)
            comm.Allreduce([ScalP_temp, MPI.DOUBLE],
                           [ScalP, MPI.DOUBLE], op=MPI.SUM)

            # 更新残差 r = r - q/(p,q)
            r_part = r_part - q_part / ScalP

            # 计算标量积 (r, r)
            ScalP_temp[0] = np.dot(r_part, r_part)
            comm.Allreduce([ScalP_temp, MPI.DOUBLE],
                           [ScalP, MPI.DOUBLE], op=MPI.SUM)

            # 更新方向向量 p = p + r/(r,r)
            p_part = p_part + r_part / ScalP

        # 收集所有进程的p_part到完整的p向量
        comm.Allgatherv([p_part, N_part, MPI.DOUBLE],
                        [p, rcounts_N, displs_N, MPI.DOUBLE])

        # 计算 q = A^T(A p)
        Ap_local = np.dot(A_part, p)
        q_temp = np.dot(A_part.T, Ap_local)

        # 归约并分散结果到各进程
        comm.Reduce_scatter([q_temp, MPI.DOUBLE],
                            [q_part, MPI.DOUBLE],
                            op=MPI.SUM)

        # 计算标量积 (p, q)
        ScalP_temp[0] = np.dot(p_part, q_part)
        comm.Allreduce([ScalP_temp, MPI.DOUBLE],
                       [ScalP, MPI.DOUBLE], op=MPI.SUM)

        # 更新解 x = x - p/(p,q)
        x_part = x_part - p_part / ScalP
        s = s + 1

    return x_part


# 辅助函数：生成数据分布数组
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


# 主程序
if __name__ == "__main__":
    # 读取问题参数
    if rank == 0:
        print("Reading problem parameters...")
        with open('in.dat', 'r') as f:
            N = int(f.readline())
            M = int(f.readline())
        print(f"N={N}, M={M}")
    else:
        N, M = 0, 0

    # 广播参数到所有进程
    N = comm.bcast(N, root=0)
    M = comm.bcast(M, root=0)

    # 生成数据分布数组
    rcounts_M, displs_M = auxiliary_arrays(M, numprocs)
    rcounts_N, displs_N = auxiliary_arrays(N, numprocs)

    # 获取本进程负责的数据大小
    M_part = rcounts_M[rank]
    N_part = rcounts_N[rank]

    # 读取并分发矩阵A
    if rank == 0:
        print("Reading matrix A...")
        A = np.empty((M, N), dtype=np.float64)
        with open('AData.dat', 'r') as f:
            for j in range(M):
                for i in range(N):
                    A[j, i] = float(f.readline())
    else:
        A = None

    # 每个进程分配自己的A_part
    A_part = np.empty((M_part, N), dtype=np.float64)

    # 使用Scatterv分发矩阵A的行
    comm.Scatterv([A, rcounts_M, displs_M, MPI.DOUBLE],
                  [A_part, MPI.DOUBLE], root=0)

    # 读取并分发向量b
    if rank == 0:
        print("Reading vector b...")
        b = np.empty(M, dtype=np.float64)
        with open('bData.dat', 'r') as f:
            for j in range(M):
                b[j] = float(f.readline())
    else:
        b = None

    # 每个进程分配自己的b_part
    b_part = np.empty(M_part, dtype=np.float64)
    comm.Scatterv([b, rcounts_M, displs_M, MPI.DOUBLE],
                  [b_part, MPI.DOUBLE], root=0)

    # 初始化解向量x并分发
    if rank == 0:
        x = np.zeros(N, dtype=np.float64)
    else:
        x = None

    x_part = np.empty(N_part, dtype=np.float64)
    comm.Scatterv([x, rcounts_N, displs_N, MPI.DOUBLE],
                  [x_part, MPI.DOUBLE], root=0)

    # 运行共轭梯度法
    if rank == 0:
        print("Starting conjugate gradient method...")

    x_part = conjugate_gradient_method(A_part, b_part, x_part, N_part,
                                       rcounts_N, displs_N, comm, N)

    # 收集结果
    if rank == 0:
        x_final = np.empty(N, dtype=np.float64)
    else:
        x_final = None

    comm.Gatherv([x_part, N_part, MPI.DOUBLE],
                 [x_final, rcounts_N, displs_N, MPI.DOUBLE], root=0)

    # 在rank 0进程输出结果
    if rank == 0:
        print("Solution found!")
        print(f"First 10 elements of solution: {x_final[:10]}")

        # 可视化结果（可选）
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