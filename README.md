# Non-Archimedean Polydisc Spaces and Applications to Optimisation

This repository contains the experiments accompanying the paper
*Non-Archimedean Polydisc Spaces and Applications to Optimisation*.

The experiments use
[`NonArchimedeanMachineLearning.jl`](https://github.com/paul-leask/NonArchimedeanMachineLearning.jl)
for the underlying spaces, models, losses, and optimizers.

## Installation

### Prerequisites

- Julia 1.10 or later.
- Access to the Julia dependencies listed in [Project.toml](Project.toml).

### First-time setup

From the repository root, install the pinned dependencies and initialise the
Julia environment:

```bash
julia --project=. -e 'using Pkg; Pkg.instantiate()'
```

Then precompile the project dependencies:

```bash
julia --project=. -e 'using Pkg; Pkg.precompile()'
```

To check that everything is ready, run the quick pipeline:

```bash
bash src/run_experiments.sh --quick
```

## Running the Experiments

Run the full paper pipeline:

```bash
bash src/run_experiments.sh
```

Run with additional Julia worker processes:

```bash
bash src/run_experiments.sh -p 4
```

The pipeline runs each experiment, computes summary statistics, and writes the
generated tables under `logs/<timestamp>/`. The `logs/latest` symlink points to
the most recent run.

Useful options include:

- `--quick` for reduced samples and epochs.
- `--epochs N` to override the number of epochs.
- `--samples N` to override samples per configuration.
- `--verbose` to include detailed per-configuration tables.
- `-p N` or `--procs N` to run with additional Julia worker processes.

## Experiments

The paper benchmarks cover:

- absolute sum minimisation,
- binary function learning,
- polynomial interpolation,
- polynomial root solving.
