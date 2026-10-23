# Evidence, provenance, and implementation status

## Supported by provided artifacts

| Claim | Evidence | Boundary |
| --- | --- | --- |
| Two CG-type MPI variants | `mpi/cg_2d_variant_{1,2}.py` | Runnable syntax; cannot certify scaling without MPI test |
| Persistent `Allreduce_init`, `Start`, `Wait` | Both two-dimensional MPI scripts | Implementation demonstrated; isolated speedup **not measured** in supplied materials |
| Cross approximation MPI prototype | `mpi/parallel_cross_approximation.py` | Research implementation; not a new algorithm invention |
| Personalized May 2026 efficiency comparison | Original three PNGs; `performance/efficiency_by_process_count.py`; May slides | The values arise from a **parameterized cost model**, not MPI wall-clock data |
| λ-PCG / PCGNR serial notebook | `notebooks/preconditioned_cgnr_inverse.ipynb` | Synthetic model with notebook-specific parameters |
| Ongoing PCGR research | User's ongoing-project description + PCGNR groundwork | No new separately validated PCGR MPI implementation uploaded |
| Polynomial inverse reconstruction | `notebooks/polynomial_expansion_inverse.ipynb` | Numerical notebook, not an established benchmark |

## Efficiency plots

Original PNGs were supplied by LI WEIZHI. Plots correspond to matrix sizes (M,N)=(200,160),(2000,1600),(20000,16000); iteration count s=16; process grid side lengths 2,3,4,5,8,16,32,48,64, equivalent to p=4,9,16,25,64,256,1024,2304,4096. The supplied `E2.py` currently uses (M,N)=(20000,16000) but supports all cases by changing these parameters; the same cost coefficients reproduce the original pictures.

Fixed coefficients: C1=1e-9, C1_tilde=1e-8, C2=1e-10. Values in `performance/data/may2026_modeled_efficiency.csv` were computed using the supplied E2.py functions with only M,N changed; original pictures have been copied unchanged.

## Reproducibility / limitations

- Full wall-clock speedup studies would need serial and parallel runtime measurements on named hardware, identical mathematical workload, and both usual `Allreduce` vs persistent `Allreduce_init` implementations.
- `Allreduce_init` availability requires MPI-4-compatible support in the runtime and mpi4py.
- `legacy/` depends on unsupplied `in.dat`, `AData.dat`, and `bData.dat`.
- `notebooks/preconditioned_cgnr_inverse.ipynb` saved output reports preconditioner rank 16 and 1 vs 14 iterations for a specific synthetic test; timings and preconditioner setup cost are not supplied.
- The May slides are attributed to the student and included as a supplemental presentation; check public-sharing permission as needed.
