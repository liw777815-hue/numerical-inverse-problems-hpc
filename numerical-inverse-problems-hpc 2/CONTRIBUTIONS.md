# Personal Research Contributions — LI WEIZHI

**Lomonosov Moscow State University · Faculty of Physics, Department of Mathematics**  
**Primary role: algorithm implementation and MPI parallelization**

## Summary

My main work has been **implementing numerical algorithms from mathematical descriptions/pseudocode and developing MPI-based parallel implementations** for experiments related to linear inverse problems and numerical linear algebra. I have focused on turning the mathematical steps into executable Python code, partitioning data among processes, and managing communication needed for distributed computation.

### 1. Conjugate-gradient implementations and MPI parallelization

- Translated conjugate-gradient-type iterative algorithms into Python implementations using numerical linear algebra operations.
- Worked on MPI-parallel versions that distribute dense matrix blocks over **two-dimensional Cartesian process grids**.
- Implemented process-group communication for local/global matrix-vector products, transposed matrix-vector products and scalar/vector reductions, using MPI communication primitives such as `Allreduce` and related collective operations in the experimental code.
- Organized row and column communicators and explored different implementations of the iterative update formulas.
- Created or worked with numerical experiments to inspect the behavior of the resulting implementations. Any claims of correctness or scaling require independent serial/parallel verification on a functioning MPI installation.

**Representative files:** `mpi/cg_2d_variant_1.py`, `mpi/cg_2d_variant_2.py`, `legacy/`.

### 2. My adaptation of the algorithms, efficiency comparison, and seminar presentation

- Took an existing mathematical research manuscript as a starting reference and **adapted the conjugate-gradient formulations to the algorithms I used**, including the regularization-related terms shown in my May 2026 seminar slides.
- Applied and adapted the computational/communication cost-model framework to compare these formulations, writing Python scripts and generating the reported numerical curves myself.
- Studied the effect of **process count, two-dimensional process topology, matrix dimensions and communication coefficients** on the calculated parallel efficiency.
- **Formulated my own conclusion** in a Russian-language seminar presentation dated **May 13, 2026**: for the configurations studied, the two variants have similar estimated parallel efficiency.
- These are **my own model-based computational results**, not borrowed graphs. The `E.py`/`E2.py` scripts calculate $T_1$ and $T_n$ from parameterized formulas; they do not record wall-clock timings of serial/MPI program runs. Thus the curves should not be called measured MPI speedup unless independently supported by timing data.

**Representative materials:** `performance/efficiency_by_topology.py`, `performance/efficiency_by_process_count.py`, `presentations/2026-05-13_cg_hpc_seminar_ru.pdf`.

### 3. Parallel cross-approximation prototype

- Worked on a Python/MPI implementation of an existing cross-approximation algorithm, including partitioned matrix data, pivot selection and low-rank factor updates.
- This is an implementation exercise/prototype; I do **not** claim to have invented the underlying cross-approximation method or demonstrated production-level speedup.

**Representative file:** `mpi/parallel_cross_approximation.py`.

### 4. Additional numerical inverse-problem experiments

- Implemented or tested methods for a one-dimensional inverse model in Jupyter notebooks.
- Explored CGNR-type iterations, regularized reference solutions, low-rank factors and numerical approximation techniques.
- These notebooks are supporting numerical experiments, while the central contribution of this portfolio is **implementation and parallelization of algorithms**.

**Representative files:** `notebooks/`.

## Attribution and scope

- **My work:** adapting algorithm formulations to the variants used, translating those algorithms into working code, developing MPI-parallel versions, evaluating the adapted efficiency model, preparing the original numerical comparisons and presenting my own conclusions.
- **Existing foundations:** conjugate-gradient methods, cross approximation, regularization formulations and the research framework originate from established literature, coursework and/or research supervision. Their mathematical invention is **not** claimed here.
- **Not currently claimed on the evidence supplied:** a new underlying numerical method, peer-reviewed publication, wall-clock-measured speedup for the `E.py`/`E2.py` curves, or validation of every MPI script in the present packaging environment. This does not rule out timing experiments conducted elsewhere.

This statement is intended to describe the work accurately for prospective supervisors. The source code and slides should be checked against actual individual authorship and any research-group publication restrictions before public release.
