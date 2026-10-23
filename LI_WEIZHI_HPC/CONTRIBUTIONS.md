# Personal Contributions · LI WEIZHI

**Affiliation:** Mathematics research group, Faculty of Physics, Lomonosov Moscow State University  
**Main responsibility:** Numerical algorithm implementation and MPI parallelization

## What I did myself

**1. Algorithms → executable Python code.** I took established mathematical/pseudocode formulations, adapted the studied CG-type variants to my research problem (including regularization-related variants in my May 2026 presentation), and implemented numerical iteration procedures. This is an implementation/adaptation contribution, not a claim to have invented CG methods.

**2. Serial → MPI parallel implementations.** I developed research scripts using dense-matrix partitioning, two-dimensional Cartesian process grids, local matrix-vector and transposed-matrix-vector operations, row/column communicators, and collective reductions. Representative code: `mpi/cg_2d_variant_1.py`, `mpi/cg_2d_variant_2.py`.

**3. Persistent MPI collective requests (`Allreduce_init`).** I implemented reusable persistent collective requests for iteration-loop reductions and used `Start()`/`Wait()` to execute them. I also experimented with `Iallreduce`. The objective is to reduce repeated setup overhead and improve communication management; **the supplied evidence does not quantify an isolated wall-clock gain caused by this change**.

**4. Own cost-model experiments and conclusions.** With an existing mathematical analysis framework as a reference, I adjusted the algorithm-specific calculations, wrote numerical scripts (`E.py`, `E2.py`), obtained the original efficiency curves for three matrix sizes, and presented my conclusion that both methods have similar **modeled parallel efficiency** in the studied cases. The May 13, 2026 slide deck records this analysis.

**5. Other algorithm implementations.** I worked on MPI cross approximation (distributed pivot selection and low-rank factors), a regularized/preconditioned CGNR (λ-PCG/PCGNR) notebook using cross approximation + QR/SVD and a 1-D inverse model, and polynomial-parameterized reconstruction.

**6. Prior accelerated iterative work.** I also studied Nesterov-type accelerated iterative reconstruction in earlier experiments. That source code is not present in the submitted file set, so it is listed as earlier work rather than as a runnable repository example.

**7. Present ongoing work.** I am continuing research on PCGR / preconditioned conjugate-gradient methods for inverse problems. The available notebook provides a **serial, regularized preconditioned CGNR proof of concept**. It is not presented as a completed MPI-PCGR package.

## Scientific result and scope

The May 2026 model experiments computed the following model-based efficiencies at a 64×64 process grid (p=4096, s=16):

| M × N | A | B |
| :--- | ---: | ---: |
| 200 × 160 | 0.9084 | 0.8894 |
| 2000 × 1600 | 0.9911 | 0.9894 |
| 20000 × 16000 | 0.9991 | 0.9989 |

These are **my own numerical outcomes from the submitted cost model**; they are not measurements of `MPI.Wtime()` and do not imply that 4096 MPI ranks were actually launched.

## Explicit boundaries

- Established mathematical methods and the original reference framework are not claimed as inventions by me.
- I do claim my own algorithm adaptation, Python/MPI implementations, model calculations, presentation and interpretation as described.
- MPI script execution and comparative wall-clock timing were not independently verified while assembling this portfolio.
- Claims about `Allreduce_init` concern its implementation and intended reduction of per-iteration communication setup, not an established measured percentage speedup.
- All results are research-stage; I do not claim a journal paper or completed PCGR method without supporting evidence.
