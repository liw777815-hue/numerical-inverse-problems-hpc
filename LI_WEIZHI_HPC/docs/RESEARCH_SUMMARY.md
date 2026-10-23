# Research implementation summary — LI WEIZHI

**Working title:** Python and MPI implementations of conjugate-gradient methods for discretized linear inverse problems  
**Affiliation:** Lomonosov Moscow State University, Faculty of Physics, Department of Mathematics  
**Research seminar:** May 13, 2026 (Russian-language slide deck included under `presentations/`)

## Problem and motivation

Discretized linear inverse problems can be represented by $Ax\approx b_\delta$, where $b_\delta$ contains measurement error. Iterative solvers are useful because they predominantly require matrix-vector products, transposed matrix-vector products and vector reductions. These operations can be implemented on distributed-memory computers, but communication overhead and numerical differences may affect parallel performance.

## Research implementation work

My primary contribution was to **adapt existing iterative algorithm formulations to the variants used in my work, implement them in Python, and develop MPI-parallel versions**. I also carried out my own efficiency comparison and presented my own conclusions. The practical tasks included:

1. Adapting two conjugate-gradient-type formulations (including regularization-related terms in the seminar formulation) and expressing them as executable numerical procedures.
2. Distributing matrix data over a two-dimensional MPI Cartesian process grid and organizing row/column communicators.
3. Implementing local matrix-vector operations and collective reductions needed to construct global iterative quantities.
4. Using and adapting a computation/communication cost-model framework, writing my own scripts to calculate parallel efficiency, and obtaining my own comparison plots and conclusions.
5. Preparing and presenting a Russian-language seminar report on the comparison.

Additional code in this repository explores MPI cross approximation and one-dimensional inverse-problem reconstructions. These are related implementation experiments rather than evidence of a new underlying mathematical method.

## Parallel-efficiency model

The seminar compares algorithm variants using the standard speedup and efficiency definitions

$$S_p = \frac{T_1}{T_p}, \qquad E_p = \frac{T_1}{p\,T_p},$$

where $T_1$ is serial runtime, $T_p$ is runtime with $p$ processes. Formula-based scripts represent local arithmetic work and MPI communication through tunable coefficients and process-grid dimensions. For the studied parameter choices, **my computed comparison** showed similar efficiency estimates for the two algorithm variants.

**Important distinction:** these curves are evaluations of an analytical cost model. They are **not direct measurements** of runtime on a supercomputer and are not evidence of a verified speedup.

## Current state and next validation steps

The repository provides original experimental code and selected numerical notebook outputs. The Jupyter notebooks were successfully re-executed while organizing the repository; the MPI scripts were not run here because an MPI runtime was unavailable. Useful next steps are to test deterministic inputs, compare serial and MPI results and residual norms, and record wall-clock times for several process counts and process-grid shapes.

## Individual contribution and supervision

I was primarily responsible for **adapting the studied algorithm formulations, coding the numerical algorithms, developing their MPI parallelizations, calculating the efficiency comparisons and presenting my conclusions**. The starting manuscript and underlying established methods are credited as background; I do not claim invention of the underlying CG method or of the entire original research framework. The accompanying slides are a presentation of the work, not a peer-reviewed publication.

**Supporting files:** [`CONTRIBUTIONS.md`](../CONTRIBUTIONS.md), [`README.md`](../README.md), [seminar slides (Russian)](../presentations/2026-05-13_cg_hpc_seminar_ru.pdf).
