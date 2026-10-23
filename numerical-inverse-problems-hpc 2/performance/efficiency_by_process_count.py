# Ли Вэйчжи
import numpy as np
import matplotlib.pyplot as plt


# --- 保持算法公式函数不变 ---
def get_Tn_A(M, N, n1, n2, s, C1, C1_tilde, C2):
    n = n1 * n2
    m_loc, n_loc = M / n1, N / n2
    c2_eff = C2
    ln1 = np.log2(n1) if (n1 > 1) else 0
    ln2 = np.log2(n2) if (n2 > 1) else 0

    t_s1 = (C1 * (6 * M * N / n + 4 * m_loc + 8 * n_loc) +
            C1_tilde * (4 * M * N / n - 2 * n_loc) +
            c2_eff * (3 * m_loc * ln2 + 2 * n_loc * ln1 + ln1 + 2 * ln2))
    t_stable = (C1 * (4 * M * N / n + 3 * m_loc + 12 * n_loc) +
                C1_tilde * (2 * M * N / n - n_loc) +
                c2_eff * (2 * m_loc * ln2 + n_loc * ln1 + ln1 + 3 * ln2))
    return  t_s1 + (s - 1) * t_stable


def get_Tn_B(M, N, n1, n2, s, C1, C1_tilde, C2):
    n = n1 * n2
    m_loc, n_loc = M / n1, N / n2
    c2_eff = C2
    ln1 = np.log2(n1) if (n1 > 1) else 0
    ln2 = np.log2(n2) if (n2 > 1) else 0
    t_init = (C1 * (2 * M * N / n + m_loc + n_loc) +
              C1_tilde * (2 * M * N / n - n_loc) +
              c2_eff * (m_loc * ln2 + n_loc * ln1))
    t_iter = (C1 * (4 * M * N / n + 7 * m_loc + 8 * n_loc + 2) +
              C1_tilde * (2 * M * N / n - n_loc) +
              c2_eff * (2 * m_loc * ln2 + n_loc * ln1 + 2 * ln1 + ln2))
    return t_init + s * t_iter


# --- 实验参数 ---
M, N, s = 20000, 16000, 16
C1, C1_tilde, C2 = 1e-9, 1e-8, 1e-10

# 1. 计算串行时间 T1 (n=1, C2=0)
T1_A = get_Tn_A(M, N, 1, 1, s, C1, C1_tilde, C2)
T1_B = get_Tn_B(M, N, 1, 1, s, C1, C1_tilde, C2)


side_list = [2, 3, 4, 5, 8, 16, 32, 48, 64]
n_values = [s * s for s in side_list]  # 计算总进程数 n

Ea_list = []
Eb_list = []

for side in side_list:
    n1 = side
    n2 = side
    n_total = n1 * n2

    # 直接计算该正方形拓扑下的 Tn
    tn_a = get_Tn_A(M, N, n1, n2, s, C1, C1_tilde, C2)
    tn_b = get_Tn_B(M, N, n1, n2, s, C1, C1_tilde, C2)

    # 计算效率 E
    Ea_list.append((T1_A / tn_a) / n_total)
    Eb_list.append((T1_B / tn_b) / n_total)


plt.figure(figsize=(12, 7))
ax = plt.gca()
ax.set_facecolor('#f4f4f4')


plt.plot(n_values, Ea_list, color='red', linestyle='-', marker='o',
         linewidth=2, markersize=8, label='Algorithm A Efficiency')
plt.plot(n_values, Eb_list, color='blue', linestyle='--', marker='s',
         linewidth=2, markersize=8, label='Algorithm B Efficiency')

# 标注网格形状 (2x2, 4x4...)
for i, side in enumerate(side_list):
    plt.text(n_values[i], Ea_list[i] + 0.02, f'{side}x{side}',
             ha='center', fontsize=9, color='red', fontweight='bold')


# 设置
plt.title(f'\n(M={M}, N={N}, s={s})',
          fontsize=15, fontweight='bold')
plt.xlabel(' n (n1 = n2)', fontsize=12)
plt.ylabel(' E', fontsize=12)

plt.xscale('log')
plt.xticks(n_values, [f'{n}' for n in n_values])

plt.ylim(0, 1.15)
plt.grid(True, which='both', linestyle=':', alpha=0.6)
plt.legend(loc='lower left', fontsize=11)


plt.tight_layout()
plt.show()
