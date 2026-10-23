"""Reproduce cost-model comparisons from the submitted E2.py formulations.

These are predictions under a cost model, NOT MPI timing measurements.
This helper was added when arranging the research portfolio, and does not
modify or replace the user's three original figures.
"""
from pathlib import Path
import csv
import numpy as np
import matplotlib.pyplot as plt

# Mathematical expressions transcribed from the provided E2.py source.
def get_Tn_A(M, N, n1, n2, s, C1, C1_tilde, C2):
    n = n1*n2
    m_loc,n_loc = M/n1,N/n2
    ln1,ln2 = np.log2(n1) if n1 > 1 else 0, np.log2(n2) if n2 > 1 else 0
    t_s1 = (C1*(6*M*N/n+4*m_loc+8*n_loc) + C1_tilde*(4*M*N/n-2*n_loc)
            + C2*(3*m_loc*ln2+2*n_loc*ln1+ln1+2*ln2))
    t_stable = (C1*(4*M*N/n+3*m_loc+12*n_loc) + C1_tilde*(2*M*N/n-n_loc)
                + C2*(2*m_loc*ln2+n_loc*ln1+ln1+3*ln2))
    return t_s1+(s-1)*t_stable

def get_Tn_B(M, N, n1, n2, s, C1, C1_tilde, C2):
    n=n1*n2
    m_loc,n_loc=M/n1,N/n2
    ln1,ln2=np.log2(n1) if n1>1 else 0,np.log2(n2) if n2>1 else 0
    t_init=(C1*(2*M*N/n+m_loc+n_loc)+C1_tilde*(2*M*N/n-n_loc)
            +C2*(m_loc*ln2+n_loc*ln1))
    t_iter=(C1*(4*M*N/n+7*m_loc+8*n_loc+2)+C1_tilde*(2*M*N/n-n_loc)
            +C2*(2*m_loc*ln2+n_loc*ln1+2*ln1+ln2))
    return t_init+s*t_iter

S=(2,3,4,5,8,16,32,48,64)
CA,CB,CC=1e-9,1e-8,1e-10
out=Path(__file__).resolve().parent/'data'
out.mkdir(exist_ok=True)
rows=[]
for m,n in [(200,160),(2000,1600),(20000,16000)]:
    t1a=get_Tn_A(m,n,1,1,16,CA,CB,CC)
    t1b=get_Tn_B(m,n,1,1,16,CA,CB,CC)
    eA,eB=[],[]
    for s in S:
        p=s*s
        a=t1a/get_Tn_A(m,n,s,s,16,CA,CB,CC)/p
        b=t1b/get_Tn_B(m,n,s,s,16,CA,CB,CC)/p
        rows.append((m,n,16,s,p,a,b));eA.append(a);eB.append(b)
    plt.figure(figsize=(9.5,5))
    plt.plot([s*s for s in S],eA,'o-',label='Algorithm A — modeled')
    plt.plot([s*s for s in S],eB,'s--',label='Algorithm B — modeled')
    plt.xscale('log');plt.xticks([s*s for s in S],[str(s*s) for s in S])
    plt.ylim(0,1.05);plt.xlabel('Modeled process count (square grid)')
    plt.ylabel('Efficiency from cost model')
    plt.title(f'M={m}, N={n}, iterations=16 (not wall-clock measurement)')
    plt.legend();plt.grid(True,alpha=0.25);plt.tight_layout()
    plt.savefig(out/f'regenerated_model_M{m}_N{n}.png',dpi=150)
    plt.close()
with open(out/'may2026_modeled_efficiency.csv','w',newline='') as f:
    writer=csv.writer(f);writer.writerow(['M','N','iterations','grid_side','modeled_processes','efficiency_algorithm_A','efficiency_algorithm_B']); writer.writerows(rows)
print('Generated model-based results (not MPI timings) in',out)
