import numpy as np
import matplotlib.pyplot as plt


def get_Tn_A(M, N, n1, n2, s, C1, C1_tilde, C2, use_comm=True):
    n = n1 * n2
    m_loc, n_loc = M / n1, N / n2
    # 统一控制通信系数
    c2_eff = C2 if use_comm else 0
    ln1 = np.log2(n1) if (n1 > 1) else 0
    ln2 = np.log2(n2) if (n2 > 1) else 0

    # 1. 修正：加上初始化项 (b.copy)
    t_init = C1 * m_loc

    # 2. 严格按照 Tn,A 公式
    t_s1 = (C1 * (6 * M * N / n + 4 * m_loc + 8 * n_loc) +
            C1_tilde * (4 * M * N / n - 2 * n_loc) +
            c2_eff * (3 * m_loc * ln2 + 2 * n_loc * ln1 + ln1 + 2 * ln2))

    t_stable = (C1 * (4 * M * N / n + 3 * m_loc + 12 * n_loc) +
                C1_tilde * (2 * M * N / n - n_loc) +
                c2_eff * (2 * m_loc * ln2 + n_loc * ln1 + ln1 + 3 * ln2))

    return t_init + t_s1 + (s - 1) * t_stable


def get_Tn_B(M, N, n1, n2, s, C1, C1_tilde, C2, use_comm=True):
    n = n1 * n2
    m_loc, n_loc = M / n1, N / n2
    c2_eff = C2 if use_comm else 0
    ln1 = np.log2(n1) if (n1 > 1) else 0
    ln2 = np.log2(n2) if (n2 > 1) else 0

    # 严格按照 Tn,B 公式
    t_init = (C1 * (2 * M * N / n + m_loc + n_loc) +
              C1_tilde * (2 * M * N / n - n_loc) +
              c2_eff * (m_loc * ln2 + n_loc * ln1))

    t_iter = (C1 * (4 * M * N / n + 7 * m_loc + 8 * n_loc + 2) +
              C1_tilde * (2 * M * N / n - n_loc) +
              c2_eff * (2 * m_loc * ln2 + n_loc * ln1 + 2 * ln1 + ln2))

    return t_init + s * t_iter


# --- 实验参数 ---
M, N, s = 20000, 10000, 16  # 使用 2000x1000，计算量足以抵消一部分通信
n_total = 64
C1, C1_tilde, C2 = 1e-9, 1.2e-9, 1e-5

# 1. 计算串行时间 T1 (统一模型：n=1, C2=0)
T1_A = get_Tn_A(M, N, 1, 1, s, C1, C1_tilde, C2, use_comm=False)
T1_B = get_Tn_B(M, N, 1, 1, s, C1, C1_tilde, C2, use_comm=False)

topos = [1, 2, 4, 8, 16, 32, 64]
Ea_list = []
Eb_list = []

for n1 in topos:
    n2 = n_total // n1
    tn_a = get_Tn_A(M, N, n1, n2, s, C1, C1_tilde, C2, use_comm=True)
    tn_b = get_Tn_B(M, N, n1, n2, s, C1, C1_tilde, C2, use_comm=True)

    # 计算效率 E = (T1 / Tn) / n_total
    Ea_list.append((T1_A / tn_a) / n_total)
    Eb_list.append((T1_B / tn_b) / n_total)

# --- 绘图 ---
# labels = [f"{t}x{n_total // t}" for t in topos]
# plt.figure(figsize=(12, 6))
# plt.plot(labels, Ea_list, 'ro-', label='Algorithm A Efficiency')
# plt.plot(labels, Eb_list, 'bs-', label='Algorithm B Efficiency')
#
# # 理想线设为 1.0 (代表 100% 效率)
# plt.axhline(y=1.0, color='g', linestyle='--', alpha=0.5, label='Ideal Efficiency (1.0)')
# plt.axhline(y=0.5, color='gray', linestyle=':', alpha=0.5)
#
# plt.title(f'Parallel Efficiency (E) Comparison\n(M={M}, N={N}, n=64, s={s})', fontsize=14)
# plt.xlabel('Process Grid Topology (n1 x n2)', fontsize=12)
# plt.ylabel('Efficiency E', fontsize=12)
# plt.ylim(0, 1.1)  # 效率通常在 0-1 之间
# plt.legend()
# plt.grid(True, ls=':', alpha=0.6)
# plt.show()

# --- 绘图 (高对比度版本) ---
labels = [f"{t}x{n_total // t}" for t in topos]
plt.figure(figsize=(12, 7))

# 算法 A: 红色实线 + 圆形标记
plt.plot(labels, Ea_list, color='red', linestyle='-', marker='o',
         linewidth=2, markersize=8, label='Algorithm A Efficiency')

# 算法 B: 蓝色虚线 + 方形标记
plt.plot(labels, Eb_list, color='blue', linestyle='--', marker='s',
         linewidth=2, markersize=8, label='Algorithm B Efficiency')

# 理想效率线 (1.0)
plt.axhline(y=1.0, color='green', linestyle=':', alpha=0.7, label='Ideal Efficiency (1.0)')

# --- 增强图表可读性 ---
plt.title(f'Parallel Efficiency (E) Comparison\n(M={M}, N={N}, n=64, s={s}，C_1={C1},C_tilde={C1_tilde},C_2={C2})',
          fontsize=16, fontweight='bold')
plt.xlabel('Process Grid Topology (n1 x n2)', fontsize=12)
plt.ylabel('Efficiency E', fontsize=12)
plt.ylim(0, 1.1)
plt.legend(fontsize=11)
plt.grid(True, which='both', linestyle=':', linewidth=0.5)

# 旋转 x 轴标签以防重叠
plt.xticks(rotation=30)

# 添加背景色以突出线条
ax = plt.gca()
ax.set_facecolor('#f0f0f0')

plt.tight_layout() # 自动调整布局
plt.show()