# Research Implementation Portfolio — LI WEIZHI

**M.Sc., Faculty of Physics, Lomonosov Moscow State University (mathematics research group)**

## Research focus
Linear inverse problems, numerical linear algebra, iterative regularization, MPI parallel implementation, and cost-model efficiency analysis.

## My contributions
1. Translated and adapted established CG-type algorithm formulations into Python numerical implementations.
2. Built MPI implementations with 2-D Cartesian process grids, distributed matrix blocks, local operations and inter-process reductions.
3. Implemented persistent `MPI.Allreduce_init` requests and `Start`/`Wait` to reuse collective communication operations inside iteration loops. This is an implemented optimization, not yet an isolated measured speedup.
4. Adapted operation-count/communication-cost calculations, created my own numerical experiments, produced model efficiency figures and drew research conclusions.
5. Implemented an MPI cross approximation prototype; studied regularized preconditioned CGNR based on cross approximation + QR/SVD; explored polynomial inverse reconstruction.

## May 2026 results
Under a parameterized efficiency model with 16 iterations, the original submitted graphics compare Algorithms A/B for matrices 200×160, 2000×1600, 20000×16000. At a 64×64 modeled process grid, A/B efficiencies are 0.9084/0.8894, 0.9911/0.9894, 0.9991/0.9989 respectively. These are **the author's calculated results**, not MPI clock measurements.

## Current direction — PCGR
Ongoing research on preconditioned CG methods for inverse problems. A serial notebook provides a λ-PCG / regularized preconditioned CGNR experimental baseline. It includes a cross-approximation + QR/SVD low-rank preconditioner; for the included synthetic example the notebook reports 1 PCGNR iteration versus 14 without preconditioning (setup cost excluded, so not a runtime comparison). A new MPI-PCGR result is not claimed at this stage.

## Related work
MPI cross approximation, polynomial-parameterized reconstruction, CGNR and Tikhonov reference comparisons, with source code and clearly separated project status in the accompanying repository.

## Research presentation
*High-performance implementations of the conjugate gradient method for ill-posed linear inverse problems*, Moscow State University student seminar, May 13, 2026 (31 slides, Russian).
