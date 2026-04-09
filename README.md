# Hierarchical Data and Optimisation in Non-Archimedean Spaces — Experiments

This repository contains the experiments accompanying the paper *Hierarchical
Data and Optimisation in Non-Archimedean Spaces*. It is a thin wrapper around
the [`NAML`](../naml) package, which
implements the underlying spaces, models, losses, and optimizers. This repo
provides:

- runnable experiment scripts for each benchmark in the paper,
- a three-stage pipeline (run → stats → tables) that produces the LaTeX
  tables used in the paper,
- standalone scripts for the worked exampales included in the application section.

## Installation

### Prerequisites

- Julia 1.10 or later.
- A local checkout of the [`NAML`](../naml) package. This repository expects
  to find it at `../naml` relative to this repo (see
  [Project.toml](Project.toml)):

  ```
  ParentDir/
  ├── naml/                                                    # the NAML package
  └── Hierarchical-Data-and-Optimisation-in-Non-Archimedean-Spaces/   # this repo
  ```

### Setup

From the root of this repo:

```bash
julia --project=. -e 'using Pkg; Pkg.instantiate()'
```

This will pick up `NAML` from `../naml` and install all remaining
dependencies listed in [Project.toml](Project.toml).

To verify the setup, run a quick smoke test of the full pipeline:

```bash
bash src/generate_paper_tables.sh --quick
```

## Repository layout

```
src/
├── experiment_utils.jl          # Shared: CLI parsing, optimizer factory, JSON I/O
├── stats_utils.jl               # Shared: rankings, aggregate statistics
├── table_utils.jl               # Shared: LaTeX formatting, generic table generators
├── util.jl                      # Shared: p-adic problem generation, losses, data
├── test_util.jl                 # Shared: testing helpers
├── make_stats.jl                # Stage 2: raw JSON → stats JSON
├── generate_paper_tables.sh     # Pipeline orchestrator (run → stats → tables)
│
├── absolute_sum_minimization/   # Experiment: minimise Σ |fᵢ(x)|
├── function_learning/           # Experiment: learn a binary classifier via cross-entropy
├── polynomial_learning/         # Experiment: learn polynomial coefficients from (x, y)
├── polynomial_solving/          # Experiment: minimise |f(z)| with a known root
└── worked_examples/             # Small standalone illustrative scripts
```

Each experiment directory follows the same convention:

| File                | Purpose                                                  |
|---------------------|----------------------------------------------------------|
| `run_experiments.jl`| Stage 1: run the experiment, write raw per-sample JSON   |
| `generate_tables.jl`| Stage 3: read stats JSON, write LaTeX tables             |
| `config.jl`         | Default configurations                                   |
| `paper_config.jl`   | Configurations used for the paper                        |
| `util.jl`           | (optional) experiment-specific problem generation        |

## The experiments

| Directory                   | Description                                                     |
|-----------------------------|-----------------------------------------------------------------|
| [absolute_sum_minimization/](src/absolute_sum_minimization/) | Minimise `|f₁(x)| + |f₂(x)| + …` over a p-adic polydisc       |
| [function_learning/](src/function_learning/)                 | Learn a binary classifier by cross-entropy minimisation        |
| [polynomial_learning/](src/polynomial_learning/)             | Learn polynomial coefficients from `(x, y)` samples            |
| [polynomial_solving/](src/polynomial_solving/)               | Minimise `|f(z)|` for a polynomial `f` with a guaranteed root  |
| [worked_examples/](src/worked_examples/)                     | Small hand-crafted examples (`x² − 1`, a cubic sum, …)         |

All four benchmarks compare the same family of optimizers from `NAML`
(random search, best-first / greedy, MCTS and DAG-MCTS variants, HOO/DOO,
gradient-style baselines, …) on problems with inputs and parameters in a
non-Archimedean polydisc space.

## Running the experiments

The pipeline has three stages:

```
run_experiments.jl  →  *_raw.json
                            ↓
make_stats.jl       →  *_stats.json
                            ↓
generate_tables.jl  →  *.tex
```

### Full paper pipeline

Run every experiment and regenerate every LaTeX table:

```bash
bash src/generate_paper_tables.sh
```

Useful flags (all forwarded to the underlying `run_experiments.jl` calls):

| Flag                      | Meaning                                                   |
|---------------------------|-----------------------------------------------------------|
| `--quick`                 | Reduced epochs/samples — smoke test                       |
| `--epochs N`              | Override number of epochs (default: 20)                   |
| `--samples N`             | Override samples per configuration (default: 30)          |
| `--selection-mode M`      | MCTS/DAG-MCTS selection: `BestValue`, `VisitCount`, `BestLoss` |
| `--degree D`              | Override tree branching degree                            |
| `--verbose`               | Include per-configuration detailed tables                 |
| `-p N`, `--procs N`       | Launch Julia with `N` additional worker processes         |
| `--paper-optimizer-comparison` | Run the main optimizer comparison suite (default)    |
| `--paper-mcts-branching`, `--paper-dag-mcts-branching`, … | Run one of the ablation suites |

### Running a single experiment by hand

Each stage can also be invoked directly — useful when iterating on one
experiment:

```bash
# Stage 1: run (with paper configs, save raw results)
julia --project=. src/absolute_sum_minimization/run_experiments.jl \
    --paper --save --output absolute_sum_results_raw.json

# Stage 2: compute statistics
julia --project=. src/make_stats.jl \
    src/absolute_sum_minimization/absolute_sum_results_raw.json \
    --output src/absolute_sum_minimization/absolute_sum_results_stats.json

# Stage 3: generate LaTeX tables
julia --project=. src/absolute_sum_minimization/generate_tables.jl \
    src/absolute_sum_minimization/absolute_sum_results_stats.json \
    --output absolute_sum_tables.tex
```

Replace `absolute_sum_minimization` with any of `function_learning`,
`polynomial_learning`, or `polynomial_solving` to run the other experiments.

### `run_experiments.jl` flags

```
--quick              Reduced epochs (5) and samples for smoke testing
--save               Save results to JSON
--config             Use configurations from config.jl
--paper              Use paper-ready configurations from paper_config.jl
--epochs N           Override epochs (default: 20)
--samples N          Override samples per configuration
--output FILE        Override output filename
--selection-mode M   MCTS/DAG-MCTS selection mode
--degree D           Override tree branching degree
--description TEXT   Stored in JSON metadata
--git-commit HASH    Stored in JSON metadata
```

### `make_stats.jl`

```bash
julia --project=. src/make_stats.jl <raw.json> [--output stats.json]
```

It auto-detects the experiment type from the JSON metadata so that
type-specific fields (e.g. accuracy for `function_learning`) are handled
correctly.

### `generate_tables.jl`

```bash
julia --project=. src/<experiment>/generate_tables.jl <stats.json> [FLAGS]

  --output FILE   Output `.tex` filename (default: <experiment>_tables.tex)
  --stdout        Print tables to stdout instead of writing a file
  --verbose       Include per-configuration detailed tables
```

## JSON schema

### Raw JSON (`*_raw.json`)

```json
{
  "metadata": {
    "experiment_type": "absolute_sum_minimization",
    "timestamp": "...",
    "n_epochs": 20,
    "quick_mode": false,
    "optimizer_order": ["Random", "Best-First", "..."],
    "description": "",
    "git_commit": ""
  },
  "experiments": [
    {
      "config": { "name": "...", "prime": 2, "...": "..." },
      "samples": [
        {
          "sample_num": 1,
          "initial_loss": 1.23,
          "optimizers": {
            "Random": {
              "time": 0.5,
              "final_loss": 0.8,
              "losses": ["..."],
              "improvement": 0.43,
              "improvement_ratio": 0.35,
              "total_evals": 100
            }
          }
        }
      ]
    }
  ]
}
```

### Stats JSON (`*_stats.json`)

Same structure as the raw JSON, plus:

- each sample's optimizer entry is annotated with a `"rank"` field,
- each experiment gets an `"aggregate"` dict (mean / std / min / max per
  optimizer),
- a top-level `"global_ranking"` dict gives average ranks across configs.

