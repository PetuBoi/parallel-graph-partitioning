# Parallel Graph Partitioning (C++ / OpenMP / MPI)

An academic C++ project exploring parallel search for a minimum-weight cut in a weighted graph under partition-size constraints. Originally developed as coursework at CTU in Prague.

## Implementations

| File | Description |
| --- | --- |
| `src/PDP_paralel.cpp` | OpenMP task-based recursive search. |
| `src/PDP_paralel_data.cpp` | OpenMP parallelization with generated subproblems. |
| `src/PDP_MPI.cpp` | Distributed approach using MPI with OpenMP. |

The programs explore vertex assignments to two partitions, calculate cut weights, and use pruning to reduce search where possible. They are **separate variants**, not interchangeable source files for one executable.

## Requirements

- C++ compiler with OpenMP support for the OpenMP variants
- MPI toolchain (e.g. Open MPI/MPICH) for the MPI variant

Example compilation on Linux with GCC:

```bash
g++ -O2 -std=c++17 -fopenmp src/PDP_paralel.cpp -o pdp_openmp
g++ -O2 -std=c++17 -fopenmp src/PDP_paralel_data.cpp -o pdp_openmp_data
mpic++ -O2 -std=c++17 -fopenmp src/PDP_MPI.cpp -o pdp_mpi
```

The examples show *how to compile*, not verified build/test results. Review each program's `main()` for its expected input format and runtime arguments before running. MPI programs must be launched using the MPI runtime (`mpirun`/`mpiexec`) as appropriate for the local environment.

## Scope and limitations

The repository preserves the original coursework source files, without refactoring or claims of production readiness. No original input datasets, performance measurements, or automated tests are included here. It is a portfolio example of algorithm design and parallel programming, **not a verified performance benchmark**.

## License

No open-source license is granted by this repository. Please contact the author before reuse.
