# Reproducibility notes and known limitations

This is an organized research-code archive rather than a production solver.

1. **MPI experiments not executed during packaging.** The packaging environment lacks `mpi4py` and an MPI launcher. Correct execution must be checked on a machine with MPI installed.
2. **Square MPI process grid assumed.** `mpi/cg_2d_variant_1.py`, `mpi/cg_2d_variant_2.py`, and `mpi/parallel_cross_approximation.py` take `sqrt(number of processes)` as both grid dimensions without checking for non-square process counts. Run with 1, 4, 9, etc. processes until generalized.
3. **Earlier CG scripts require text data files.** `legacy/cg_1d_experiment.py` and `legacy/cg_distributed_experiment.py` reference `in.dat`, `AData.dat`, and `bData.dat`, which were not in the submitted material. They may have additional API/distribution issues; treat them as archival.
4. **Notebook fix.** Two notebook cells in `polynomial_expansion_inverse.ipynb` had unexpected top-level indentation in the uploaded source. Packaging only dedented these two cells; it did not change the numerical algorithm or parameters. The notebook was subsequently executed successfully during packaging validation; the numerical algorithm and parameters were unchanged.
5. **Saved figures.** Figures under `figures/` were extracted from the existing notebook outputs. Both notebooks separately completed an execution check in a Python environment during packaging; plots in `figures/` were not overwritten.
6. **Mathematical and performance caveat.** `performance/` evaluates a parameterized analytical model; comparisons must not be described as measured MPI speedups.
7. **Implementation inspection.** Some matrix partition/distribution and endpoint-handling logic may need additional numerical tests; do not claim all implementations are correct until comparing against serial reference solutions.
8. **Authorship and external documents.** The uploaded Russian-language theoretical report is not included because permission, authorship, and publication status were not established.

Suggested verification protocol: (a) record software/MPI versions, (b) test deterministic small matrices and synthetic right-hand sides, (c) compare serial and MPI `Ax`, `A^T b` and solver results, (d) record residual norms, (e) repeat timing across different process counts.

## Notebook validation log (packaging check)

- `preconditioned_cgnr_inverse.ipynb`: executed successfully, displayed preconditioner retained rank 16, PCGNR 1 iteration, plain CGNR 14 iterations; these are values for the provided synthetic problem and stopping rule, not general performance claims.
- `polynomial_expansion_inverse.ipynb`: executed successfully after the cell-indentation repair; SciPy optimizer returned convergence with polynomial degree parameter `k=13`.
- No MPI script was executed; MPI code was syntax-checked only.

9. **Included seminar presentation.** The May 13, 2026 Russian-language slide deck is now included as an attributed student seminar presentation. The slide figures for parallel-efficiency comparison are based on analytical cost-model calculations; they should not be labeled real MPI timing results. Review the slides for contact details and public-sharing permission before publication.
